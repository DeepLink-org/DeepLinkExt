# Copyright (c) 2024, DeepLink.

from typing import Optional, Union, List
import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")
assert hasattr(ext, "apply_rotary")
assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")
assert hasattr(ext, "adamw")
assert hasattr(ext, "scaled_masked_softmax_fwd") and hasattr(ext, "scaled_masked_softmax_bwd")


class DeepLinkFlashSelfAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attention_mask, dropout_p, softmax_scale, head_num):
        (
            out,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd_v2(
            q,
            k,
            v,
            attention_mask,
            dropout_p,
            softmax_scale,
            head_num,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.head_num = head_num
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            q,
            k,
            v,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors
        attention_mask = (
            torch.Tensor().cuda() if attention_mask is None else attention_mask
        )
        dropout_mask = torch.Tensor().cuda() if dropout_mask is None else dropout_mask
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        ext.fa_bwd(
            dq,
            dk,
            dv,
            dout,
            q,
            k,
            v,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.head_num,
        )
        return dq, dk, dv, None, None, None, None


def apply_rotary_for_ascend_speed(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    output = torch.empty_like(x)
    ext.apply_rotary(output, x, cos, sin, conjugate, interleaved)
    return output


class DeepLinkRotaryEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, cos, sin):
        ctx.save_for_backward(cos, sin)
        return apply_rotary_for_ascend_speed(t, cos, sin)

    @staticmethod
    def backward(ctx, t):
        cos, sin = ctx.saved_tensors
        return apply_rotary_for_ascend_speed(t, cos, sin, conjugate=True), None, None


class DeepLinkRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        bias = torch.Tensor().cuda()
        output, inv_rms = ext.rms_norm(hidden_states, None, weight, bias, eps)
        ctx.save_for_backward(hidden_states, inv_rms, weight, bias)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = ext.rms_norm_backward(
            hidden_states, grad_output, inv_rms, None, weight, bias, ctx.eps
        )
        return grad_input, grad_weight, None, None


def adamw_for_ascend_speed(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    norm_coeff_scale: float
):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """

    assert maximize == False, "ascend diopiAdamW only support False 'maximize'."
    assert amsgrad == False, "ascend diopiAdamW only support False 'amsgrad'."

    for i, param in enumerate(params):
        if norm_coeff_scale is not None:
            grad = grads[i].float() * norm_coeff_scale
        else:
            grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if not max_exp_avg_sqs:
            max_exp_avg_sq = torch.Tensor().cuda()
        else:
            max_exp_avg_sq = max_exp_avg_sqs[i]
        ext.adamw(
            param,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            grad,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            amsgrad,
        )
    return params, exp_avgs, exp_avg_sqs


class DeepLinkScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, scale, fixed_triu_mask):
        out = ext.scaled_masked_softmax_fwd(input, mask, scale, fixed_triu_mask)
        ctx.save_for_backward(out, mask)
        ctx.scale = scale
        ctx.fixed_triu_mask = fixed_triu_mask
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, mask = ctx.saved_tensors
        grad_input = ext.scaled_masked_softmax_bwd(
            grad_output, out, mask, ctx.scale, ctx.fixed_triu_mask
        )
        return grad_input, None, None, None
