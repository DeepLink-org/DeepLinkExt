# Copyright (c) 2024, DeepLink.

import torch
import torch_npu

__all__ = ["RotaryEmbedding"]


def _unsqueeze_to_4d(x: torch.Tensor):
    while x.dim() < 4:
        x = x.unsqueeze(0)
    return x


def apply_rotary(x: torch.Tensor, cos, sin, confj=False, interleaved=False):
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

    print(cos_cat.shape)
    print(x_view.shape)

    cos_x = torch.mul(cos_cat, x_view)
    sin_x = torch.mul(sin_cat, x_view_new)
    out = cos_x + sin_x

    return out


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
