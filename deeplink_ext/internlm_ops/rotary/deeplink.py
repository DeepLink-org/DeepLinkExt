# Copyright (c) 2024, DeepLink.

import torch
from einops import rearrange
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "apply_rotary")


class DeepLinkApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert (
            sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        )
        q_ro = qkv[:, :, 0, :, :rotary_dim]
        ext.apply_rotary(
            q_ro,
            q_ro,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            False,
            interleaved,
        )
        k_ro = qkv[:, :, 1, :, :rotary_dim]
        ext.apply_rotary(
            k_ro,
            k_ro,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            False,
            interleaved,
        )
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        interleaved = ctx.interleaved
        _, seqlen, _, _, headdim = dqkv.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq_ro = dqkv[:, :, 0, :, :rotary_dim]
        ext.apply_rotary(
            dq_ro,
            dq_ro,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            True,
            interleaved,
        )
        dk_ro = dqkv[:, :, 1, :, :rotary_dim]
        ext.apply_rotary(
            dk_ro,
            dk_ro,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            True,
            interleaved,
        )
        return dqkv, None, None, None, None, None


class DeepLinkApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]

        ext.apply_rotary(
            out_ro,
            x_ro,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            False,
            interleaved,
        )

        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        dx = torch.empty_like(do) if not inplace else do
        dx_ro = dx[..., :rotary_dim]
        ext.apply_rotary(
            dx_ro,
            do_ro,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            True,
            ctx.interleaved,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None
