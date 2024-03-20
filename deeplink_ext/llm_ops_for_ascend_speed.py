# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")


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
