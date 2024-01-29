# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")


class DeepLinkFlashAttentionKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, dropout_p, softmax_scale, causal):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_max, softmax_sum, softmax_out, rng = ext.fa_fwd(
            q, kv[:, :, 0], kv[:, :, 1], dropout_p, softmax_scale, causal
        )
        ctx.save_for_backward(
            q, kv, out, softmax_max, softmax_sum, softmax_out, rng.get_state()
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, kv, out, softmax_max, softmax_sum, softmax_out, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        ext.fa_bwd(
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            dout,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            out,
            softmax_max,
            softmax_sum,
            softmax_out,
            rng,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
        )
        return dq, dkv, None, None, None, None
