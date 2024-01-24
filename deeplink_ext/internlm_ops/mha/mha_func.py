# Copyright (c) 2023, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext


class DeepLinkMultiHeadAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_fwd(
            q,
            k,
            v,
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, rng.get_state())
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        dq, dk, dv = ext.mha_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            None,
            None,
            None,
        )
        return dq, dk, dv, None, None, None, None
