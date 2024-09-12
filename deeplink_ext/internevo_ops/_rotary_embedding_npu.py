# Copyright (c) 2024, DeepLink.

import torch
import torch_npu
from einops import rearrange

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

        re_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin[:seqlen], "s d -> s 1 d")

        cat_cos = torch.cat([re_cos, re_cos], -1)
        cat_sin = torch.cat([re_sin, re_sin], -1)

        rot = torch_npu.npu_rotary_mul(x[..., :rotary_dim], cat_cos, cat_sin)
        ctx.save_for_backward(cat_cos, cat_sin)
        ctx.interleaved = interleaved
        ctx.in_place = in_place
        if in_place:
            x[..., :rotary_dim].copy_(rot)
            return x
        else:
            out = x.detach().clone()
            if rotary_dim < head_dim and not in_place:
                out[..., rotary_dim:].copy_(x[..., rotary_dim:])
            return out

    @staticmethod
    def backward(ctx, do):
        cat_cos, cat_sin = ctx.saved_tensors
        *_, seqlen, _, head_dim = do.shape
        rotary_dim = cat_cos.shape[-1]

        dx_out = torch_npu.npu_rotary_mul(
            do[..., :rotary_dim], cat_cos, torch.neg(cat_sin)
        )
        if ctx.in_place:
            do[..., :rotary_dim].copy_(dx_out)
            return do, None, None, None, None
        else:
            dx = do.detach().clone()
            dx[..., :rotary_dim].copy_(dx_out)
            return dx, None, None, None, None
