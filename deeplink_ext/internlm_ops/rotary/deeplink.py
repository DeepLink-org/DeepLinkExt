# Copyright (c) 2024, DeepLink.

from typing import Optional, Union
import torch
from einops import rearrange
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "apply_rotary")

__all__ = ["apply_rotary"]


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
    

def apply_rotary_for_ascend_speed(
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
    output = torch.empty_like(x)
    ext.apply_rotary(
        output,
        x,
        cos,
        sin,
        conjugate,
        interleaved
    )
    return output

class RotaryEmbedding_AscendSpeed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, cos, sin):
        ctx.save_for_backward(cos, sin)
        return apply_rotary_for_ascend_speed(t, cos, sin)

    
    @staticmethod
    def backward(ctx, t):
        cos, sin = ctx.saved_tensors
        return apply_rotary_for_ascend_speed(t, cos, sin, conjugate=True), None, None