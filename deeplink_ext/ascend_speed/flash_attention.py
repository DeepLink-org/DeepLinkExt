# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "custom_fa_fwd") and hasattr(ext, "custom_fa_bwd")

__all__ = ["FlashSelfAttention"]


class FlashSelfAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, q, k, v, attention_mask, dropout_p, softmax_scale, head_num, input_layout
    ):
        out = torch.empty_like(q)
        assert (
            q.device == k.device and k.device == v.device
        ), "the devices of q, k and v are not same"
        gen = torch.Generator(device=q.device)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_fwd(
            out,
            gen,
            q,
            k,
            v,
            None,
            attention_mask,
            dropout_p,
            softmax_scale,
            attention_mask is not None,
            -1,
            -1,
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
        ext.custom_fa_bwd(
            dq,
            dk,
            dv,
            dout,
            q,
            k,
            v,
            None,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            attention_mask is not None,
            -1,
            -1,
            ctx.head_num,
            ctx.input_layout,
        )
        return dq, dk, dv, None, None, None, None, None
