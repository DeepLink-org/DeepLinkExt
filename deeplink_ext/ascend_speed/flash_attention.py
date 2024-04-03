import torch
import torch_dipu
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd_v2") and hasattr(ext, "fa_bwd")


class FlashSelfAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, attention_mask, dropout_p, softmax_scale, head_num, input_layout
    ):
        out = torch.empty_like(q)
        gen = torch_dipu._C._create_dipu_generator(-1)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd_v2(
            out,
            q,
            k,
            v,
            gen,
            attention_mask,
            dropout_p,
            softmax_scale,
            head_num,
            input_layout,
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
        ctx.input_layout = input_layout
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
            ctx.input_layout,
        )
        return dq, dk, dv, None, None, None, None, None
