# Copyright (c) 2024, DeepLink.

import torch
from einops import rearrange

__all__ = ["ApplyRotaryEmbTorch"]


def _torch_apply_rotary_func(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool = False,
):
    assert (
        x1.device == x2.device == cos.device == sin.device
    ), "All inputs must be on the same device"
    assert (
        x1.dtype == x2.dtype == cos.dtype == sin.dtype
    ), "All inputs must have the same dtype"
    assert x1.size() == x2.size(), "Input x1 and x2 must have the same sizes"
    assert cos.size() == sin.size(), "Input cos and sin must have the same sizes"

    x1, x2, cos, sin = x1.float(), x2.float(), cos.float(), sin.float()

    if conj:
        out1.copy_(x1 * cos + x2 * sin)
        out2.copy_(-x1 * sin + x2 * cos)
    else:
        out1.copy_(x1 * cos - x2 * sin)
        out2.copy_(x1 * sin + x2 * cos)


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L35
class ApplyRotaryEmbTorch(torch.autograd.Function):
    """
    ApplyRotaryEmbTorch
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

        x_ro = x[..., :rotary_dim]
        x1, x2 = (
            (x_ro[..., ::2], x_ro[..., 1::2]) if interleaved else x_ro.chunk(2, dim=-1)
        )

        if in_place:
            out, o1, o2 = x, x1, x2
        else:
            out = torch.empty_like(x)
            out_ro = out[..., :rotary_dim]
            o1, o2 = (
                (out_ro[..., ::2], out_ro[..., 1::2])
                if interleaved
                else out_ro.chunk(2, dim=-1)
            )

        _torch_apply_rotary_func(
            x1,
            x2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            o1,
            o2,
            False,
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

        do_ro = do[..., :rotary_dim]
        do1, do2 = (
            (do_ro[..., ::2], do_ro[..., 1::2])
            if ctx.interleaved
            else do_ro.chunk(2, dim=-1)
        )

        if ctx.in_place:
            dx, dx1, dx2 = do, do1, do2
        else:
            dx = torch.empty_like(do)
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (
                (dx_ro[..., ::2], dx_ro[..., 1::2])
                if ctx.interleaved
                else dx_ro.chunk(2, dim=-1)
            )

        _torch_apply_rotary_func(
            do1,
            do2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            dx1,
            dx2,
            True,
        )

        if rotary_dim < head_dim and not ctx.in_place:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])

        return dx, None, None, None, None
