import torch
from typing import Optional, Union
import deeplink_ext.cpp_extensions as ext


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
    ext.apply_rotary(output, x, cos, sin, conjugate, interleaved)
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
