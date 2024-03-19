# Copyright (c) 2023, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_varlen_fwd") and hasattr(ext, "fa_varlen_bwd")


class DeepLinkFlashAttentionVarLenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        (
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_varlen_fwd(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            cu_seqlens[1:].to(torch.int64),
            cu_seqlens[1:].to(torch.int64),
            dropout_p,
            softmax_scale,
            causal,
        )
        ctx.save_for_backward(
            qkv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            cu_seqlens,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            qkv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            cu_seqlens,
        ) = ctx.saved_tensors
        attention_mask = (
            torch.Tensor().cuda() if attention_mask is None else attention_mask
        )
        dropout_mask = torch.Tensor().cuda() if dropout_mask is None else dropout_mask
        dqkv = torch.empty_like(qkv)
        ext.fa_varlen_bwd(
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            dout,
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            out,
            cu_seqlens[1:],
            cu_seqlens[1:],
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
        )
        return dqkv, None, None, None, None, None, None, None, None
