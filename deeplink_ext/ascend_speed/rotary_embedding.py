# Copyright (c) 2024, DeepLink.

import torch
from typing import Optional, Union
import deeplink_ext.cpp_extensions as ext


__all__ = ["RotaryEmbedding"]


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved=False,
    conjugate=False,
) -> torch.Tensor:
    output = torch.empty_like(x)
    ext.apply_rotary(output, x, cos, sin, conjugate, interleaved)
    return output


class RotaryEmbedding(torch.autograd.Function):
    """
    Apply rotary positional embedding to input tensor x.
    Args:
        x (Tensor): Input tensor x is of shape [seq_length, ... , dim]
        cos (Tensor): Input tensor cos is of shape [seq_length, ..., dim]
        sin (Tensor): Input tensor sin is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        cos, _ = torch.chunk(cos, 2, -1)
        sin, _ = torch.chunk(sin, 2, -1)
        ctx.save_for_backward(cos, sin)
        return apply_rotary(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        return apply_rotary(grad_output, cos, sin, conjugate=True), None, None
