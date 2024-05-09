# Copyright (c) 2024, DeepLink.

import torch


__all__ = ["RotaryEmbeddingTorch"]


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    conjugate=False,
) -> torch.Tensor:
    x, cos, sin = x.float(), cos.float(), sin.float()
    if conjugate:
        torch.neg_(sin)
    output = x * cos + _rotate_half(x) * sin
    return output


class RotaryEmbeddingTorch(torch.autograd.Function):
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
        ctx.save_for_backward(cos, sin)
        return apply_rotary_torch(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        return apply_rotary_torch(grad_output, cos, sin, conjugate=True), None, None
