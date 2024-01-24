# Copyright (c) 2023, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext


class DeepLinkMultiHeadAttentionVarLenKVPackedFunc(torch.autograd.Function):
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
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_varlen_fwd(
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(
            q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng.get_state()
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen_q
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        (
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = ctx.saved_tensors
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        ext.mha_varlen_bwd(
            dout,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
        )
        return dq, dkv, None, None, None, None, None, None, None, None
