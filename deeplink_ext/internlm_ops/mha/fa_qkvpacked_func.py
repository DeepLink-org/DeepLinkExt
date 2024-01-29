# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")


class DeepLinkFlashAttentionQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, dropout_p, softmax_scale, causal):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_max, softmax_sum, softmax_out, rng = ext.fa_fwd(
            qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], dropout_p, softmax_scale, causal
        )
        ctx.save_for_backward(
            qkv, out, softmax_max, softmax_sum, softmax_out, rng.get_state()
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        qkv, out, softmax_max, softmax_sum, softmax_out, rng_state = ctx.saved_tensors
        dqkv = torch.empty_like(qkv)
        rng = torch.Generator(device=qkv.device)
        rng.set_state(rng_state)
        ext.fa_bwd(
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            dout,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            out,
            softmax_max,
            softmax_sum,
            softmax_out,
            rng,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
        )
        return dqkv, None, None, None, None
