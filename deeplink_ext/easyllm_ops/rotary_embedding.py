# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext


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
    # 如果使用累积序列长度cu_seqlens，则使用变长模式，但目前由于kernel的限制，暂无设备支持变长模式
    is_varlen = cu_seqlens is not None
    assert not is_varlen, "varlen mode rotary embedding not supported yet."
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
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

    cos, sin = cos.contiguous(), sin.contiguous()
    # if isinstance(seqlen_offsets, torch.Tensor):
    #     assert seqlen_offsets.shape == (batch,)
    #     assert seqlen_offsets.dtype in [torch.int32, torch.int64]
    #     seqlen_offsets = seqlen_offsets.contiguous()
    # else:
    #     assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim:
        if not inplace:
            output[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ext.apply_rotary(
            output[..., :rotary_dim],
            x[..., :rotary_dim],
            cos,
            sin,
            conjugate,
            interleaved,
        )
    else:
        ext.apply_rotary(
            output,
            x,
            cos,
            sin,
            conjugate,
            interleaved,
        )
    return output
