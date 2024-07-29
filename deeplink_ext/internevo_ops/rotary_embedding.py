# Copyright (c) 2024, DeepLink.

import torch
from einops import rearrange
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "apply_rotary")

__all__ = ["ApplyRotaryEmb"]


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L35
class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        in_place: bool = False,
    ):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        *_, seqlen, _, head_dim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2

        assert rotary_dim <= head_dim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)

        if in_place:
            out = x
        else:
            out = torch.empty_like(x)

        ext.apply_rotary(
            out[..., :rotary_dim],
            x[..., :rotary_dim],
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            False,
            interleaved,
        )

        if rotary_dim < head_dim and not in_place:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])

        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.in_place = in_place

        return out

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        *_, seqlen, _, head_dim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2

        if ctx.in_place:
            dx = do
        else:
            dx = torch.empty_like(do)

        ext.apply_rotary(
            dx[..., :rotary_dim],
            do[..., :rotary_dim],
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            True,
            ctx.interleaved,
        )

        if rotary_dim < head_dim and not ctx.in_place:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])

        return dx, None, None, None, None
