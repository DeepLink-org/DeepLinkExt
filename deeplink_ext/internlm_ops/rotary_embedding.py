# Copyright (c) 2024, DeepLink.

from typing import Optional, Union
import torch
from einops import rearrange
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "apply_rotary")

__all__ = ["apply_rotary", "DeeplinkApplyRotaryEmb", "DeeplinkApplyRotaryEmbQKV_"]


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    if seqlen_offsets != 0 and cu_seqlens is None and max_seqlen is None:
        raise NotImplementedError(
            "apply_rotary: seqlen_offsets, cu_seqlens and max_seqlen are not supported yet"
        )
    batch, seqlen, nheads, headdim = x.shape
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"
    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    ext.apply_rotary(
        output,
        x,
        rearrange(cos[:seqlen], "s d -> s 1 d"),
        rearrange(sin[:seqlen], "s d -> s 1 d"),
        conjugate,
        interleaved,
    )
    return output


class DeeplinkApplyRotaryEmb(torch.autograd.Function):
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
        new_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
        new_sin = rearrange(sin[:seqlen], "s d -> s 1 d")
        if rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(new_cos, new_sin)
        ctx.interleaved = interleaved
        ext.apply_rotary(
            out[..., :rotary_dim],
            x[..., :rotary_dim],
            new_cos,
            new_sin,
            False,
            interleaved,
        )
        return out

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dx = torch.empty_like(do)
        ext.apply_rotary(
            dx[..., :rotary_dim],
            do[..., :rotary_dim],
            cos,
            sin,
            True,
            ctx.interleaved,
        )
        if rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


class DeeplinkApplyRotaryEmbQKV_(torch.autograd.Function):
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
        out = torch.empty_like(q_ro)
        ext.apply_rotary(
            out,
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
        out = torch.empty_like(k_ro)
        ext.apply_rotary(
            out,
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
        interleaved = ctx.interleaved
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        seqlen = None if len(dqkv.shape) == 4 else dqkv.shape[1]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq_ro = (
            dqkv[:, 0, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 0, :, :rotary_dim]
        )
        out = torch.empty_like(dq_ro)
        ext.apply_rotary(
            out,
            dq_ro,
            cos,
            sin,
            True,
            interleaved,
        )
        dq_ro.copy_(out)

        dk_ro = (
            dqkv[:, 1, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 1, :, :rotary_dim]
        )
        out = torch.empty_like(dk_ro)
        ext.apply_rotary(
            out,
            dk_ro,
            cos_k,
            sin_k,
            True,
            interleaved,
        )
        dk_ro.copy_(out)
        return dqkv, None, None, None, None, None
