# Copyright (c) 2023, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_varlen_fwd") and hasattr(ext, "fa_varlen_bwd")


class DeepLinkFlashAttentionVarLenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        (
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_varlen_fwd(
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            cu_seqlens_q[1:],
            cu_seqlens_k[1:],
            dropout_p,
            softmax_scale,
            causal,
        )
        ctx.save_for_backward(
            q,
            kv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            q,
            kv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors
        attention_mask = (
            torch.Tensor().cuda() if attention_mask is None else attention_mask
        )
        dropout_mask = torch.Tensor().cuda() if dropout_mask is None else dropout_mask
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        ext.fa_varlen_bwd(
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            dout,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            cu_seqlens_q[1:],
            cu_seqlens_k[1:],
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
        )
        return dq, dkv, None, None, None, None, None, None, None, None
