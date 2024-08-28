# Copyright (c) 2024, DeepLink.
# Copyright (c) 2024, InternEvo.

import torch
import torch_npu
from einops import rearrange

__all__ = ["ApplyRotaryEmb", "ApplyRotaryEmbQKV_"]


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


class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        _, seqlen, _, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        out = torch.empty_like(x)
        re_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin[:seqlen], "s d -> s 1 d")
        out = _apply_rotary(
            x[..., :rotary_dim],
            re_cos,
            re_sin,
            False,
            interleaved,
        )
        if rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(re_cos, re_sin)
        ctx.interleaved = interleaved
        return out

    @staticmethod
    def backward(ctx, do):
        re_cos, re_sin = ctx.saved_tensors
        headdim = do.shape[-1]
        rotary_dim = re_cos.shape[-1]
        rotary_dim *= 2
        dx = _apply_rotary(
            do[..., :rotary_dim],
            re_cos,
            re_sin,
            True,
            ctx.interleaved,
        )
        if rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        """
            qkv: (total, 3, nheads, headdim) / (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        # len(qkv.shape) == 4 means the format of qkv is (total, 3, nheads, headdim) which is packed,
        # otherwise the format of qkv is (batch_size, seqlen, 3, nheads, headdim) which is unpacked.
        # We handle both packed qkv and unpacked qkv scenario in this class.
        three = qkv.shape[1] if len(qkv.shape) == 4 else qkv.shape[2]
        assert three == 3
        seqlen = None if len(qkv.shape) == 4 else qkv.shape[1]
        rotary_seqlen, rotary_dim = cos.shape
        if len(qkv.shape) != 4:
            assert seqlen <= rotary_seqlen
        headdim = qkv.shape[-1]
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert (
            sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        )

        q_ro = (
            qkv[:, 0, :, :rotary_dim]
            if len(qkv.shape) == 4
            else qkv[:, :, 0, :, :rotary_dim]
        )
        re_cos = (
            rearrange(cos, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(cos[:seqlen], "s d -> s 1 d")
        )
        re_sin = (
            rearrange(sin, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(sin[:seqlen], "s d -> s 1 d")
        )

        # qro
        out = _apply_rotary(
            q_ro,
            re_cos,
            re_sin,
            False,
            interleaved,
        )
        q_ro.copy_(out)

        k_ro = (
            qkv[:, 1, :, :rotary_dim]
            if len(qkv.shape) == 4
            else qkv[:, :, 1, :, :rotary_dim]
        )
        re_cos_k = (
            rearrange(cos_k, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(cos_k[:seqlen], "s d -> s 1 d")
        )
        re_sin_k = (
            rearrange(sin_k, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(sin_k[:seqlen], "s d -> s 1 d")
        )
        out = _apply_rotary(
            k_ro,
            re_cos_k,
            re_sin_k,
            False,
            interleaved,
        )
        k_ro.copy_(out)

        ctx.save_for_backward(re_cos, re_sin, re_cos_k, re_sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        re_cos, re_sin, re_cos_k, re_sin_k = ctx.saved_tensors
        rotary_dim = re_cos.shape[-1]
        rotary_dim *= 2

        dq_ro = (
            dqkv[:, 0, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 0, :, :rotary_dim]
        )
        out = _apply_rotary(
            dq_ro,
            re_cos,
            re_sin,
            True,
            ctx.interleaved,
        )
        dq_ro.copy_(out)

        dk_ro = (
            dqkv[:, 1, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 1, :, :rotary_dim]
        )
        out = _apply_rotary(
            dk_ro,
            re_cos_k,
            re_sin_k,
            True,
            ctx.interleaved,
        )
        dk_ro.copy_(out)
        return dqkv, None, None, None, None, None
