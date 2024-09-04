# Copyright (c) 2024, DeepLink.

import torch
import torch_npu

__all__ = ["RotaryEmbedding"]


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
        out = torch_npu.npu_rotary_mul(x, cos, sin)
        ctx.save_for_backward(out, cos, sin)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, cos, sin = ctx.saved_tensors
        return torch_npu.npu_rotary_mul_backward(grad_output, out, cos, sin)[0], None, None
