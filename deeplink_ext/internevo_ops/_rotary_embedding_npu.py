# Copyright (c) 2024, DeepLink.

import torch
import torch_npu
from einops import rearrange

__all__ = ["ApplyRotaryEmb"]


def _unsqueeze_to_4d(x: torch.Tensor):
    while x.dim() < 4:
        x = x.unsqueeze(0)
    return x


def _apply_rotary(x: torch.Tensor, cos, sin, confj, interleaved):
    assert interleaved == False, "interleaved not support by torch_npu"

    x_view = _unsqueeze_to_4d(x)
    cos_view = _unsqueeze_to_4d(cos)
    sin_view = _unsqueeze_to_4d(sin)

    cos_cat = torch.cat([cos_view, cos_view], -1)
    sin_cat = torch.cat([sin_view, sin_view], -1)

    if confj:
        sin_cat.neg_()

    x_view_chunks = x_view.chunk(2, -1)
    x_view_new = torch.cat([-x_view_chunks[1], x_view_chunks[0]], -1)

    cos_x = torch.mul(cos_cat, x_view)
    sin_x = torch.mul(sin_cat, x_view_new)
    out = cos_x + sin_x

    return out


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

        out = _apply_rotary(x[..., :rotary_dim], rearrange(cos[:seqlen], "s d -> s 1 d"),
                            rearrange(sin[:seqlen], "s d -> s 1 d"), False, interleaved)

        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.in_place = in_place

        if in_place:
            x[..., :rotary_dim].copy_(out[..., :rotary_dim])
            return x
        else:
            if rotary_dim < head_dim:
                out[..., rotary_dim:].copy_(x[..., rotary_dim:])
            return out

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        *_, seqlen, _, head_dim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2

        out = _apply_rotary(
            do[..., :rotary_dim],
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            True,
            ctx.interleaved,
        )

        if ctx.in_place:
            do[..., :rotary_dim].copy_(out[..., :rotary_dim])
            return do, None, None, None, None
        else:
            if rotary_dim < head_dim:
                out[..., rotary_dim:].copy(do[..., rotary_dim:])
            return out, None, None, None, None
