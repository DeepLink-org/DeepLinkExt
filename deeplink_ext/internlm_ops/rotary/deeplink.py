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
        assert (
            sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim)
        )
        # TODO: better method to check whether cos/sin is half or full length
        if rotary_dim * 2 <= headdim:
            rotary_dim = rotary_dim * 2
        assert (rotary_dim <= headdim or rotary_dim * 2 <= headdim)
        assert seqlen <= rotary_seqlen
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        qk_ro = qkv[:, :, :2, :, :rotary_dim].view([batch, seqlen, 2 * nheads, headdim])
        cos_qk = None
        sin_qk = None
        if seqlen == rotary_seqlen:
            cos_qk = cos
            sin_qk = sin
        else: # <
            cos_qk = rearrange(cos[:seqlen], "s d -> s 1 d")
            sin_qk = rearrange(sin[:seqlen], "s d -> s 1 d")
        ext.apply_rotary(
            qk_ro,
            qk_ro,
            cos_qk,
            sin_qk,
            False,
            interleaved,
        )
        ctx.save_for_backward(cos_qk, sin_qk, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos_qk, sin_qk, cos_k, sin_k = ctx.saved_tensors
        interleaved = ctx.interleaved
        batch, seqlen, three, nheads, headdim = dqkv.shape
        rotary_dim = cos_qk.size(-1)
        # TODO: better method to check whether cos/sin is half or full length
        if rotary_dim * 2 <= headdim:
            rotary_dim = rotary_dim * 2
        dqk_ro = dqkv[:, :, :2, :, :rotary_dim].view([batch, seqlen, 2 * nheads, headdim])
        ext.apply_rotary(
            dqk_ro,
            dqk_ro,
            cos_qk,
            sin_qk,
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
        assert sin.shape == (rotary_seqlen, rotary_dim)
        # TODO: better method to check whether cos/sin is half or full length
        if rotary_dim * 2 <= headdim:
            rotary_dim = rotary_dim * 2
        assert (rotary_dim <= headdim or rotary_dim * 2 <= headdim)
        assert seqlen <= rotary_seqlen
        x_ro = x[..., :rotary_dim]
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]

        cos_qk = None
        sin_qk = None
        if seqlen == rotary_seqlen:
            cos_qk = cos
            sin_qk = sin
        else: # <
            cos_qk = rearrange(cos[:seqlen], "s d -> s 1 d")
            sin_qk = rearrange(sin[:seqlen], "s d -> s 1 d")

        ext.apply_rotary(
            out_ro,
            x_ro,
            cos_qk,
            sin_qk,
            False,
            interleaved,
        )

        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos_qk, sin_qk)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos_qk, sin_qk = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos_qk.size(-1)
        # TODO: better method to check whether cos/sin is half or full length
        if rotary_dim * 2 <= headdim:
            rotary_dim = rotary_dim * 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        dx = torch.empty_like(do) if not inplace else do
        dx_ro = dx[..., :rotary_dim]
        ext.apply_rotary(
            dx_ro,
            do_ro,
            cos_qk,
            sin_qk,
            True,
            ctx.interleaved,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None
