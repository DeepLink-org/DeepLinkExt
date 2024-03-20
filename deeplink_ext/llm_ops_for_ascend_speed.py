# Copyright (c) 2024, DeepLink.

from typing import Optional, Union
import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")
assert hasattr(ext, "apply_rotary")
assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")


class DeepLinkFlashSelfAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, head_num):
        (
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
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
    def forward(ctx, hidden_states, weight, bias, eps):
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
        return grad_input, grad_weight, grad_bias, None
