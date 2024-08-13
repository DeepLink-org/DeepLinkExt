# Copyright (c) 2024, DeepLink.

import torch
import torch_npu
from torch_npu import npu_rms_norm, npu_rms_norm_backward

__all__ = ["RMSNorm"]


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        output, inv_rms = npu_rms_norm(hidden_states, weight, eps)
        ctx.save_for_backward(hidden_states, inv_rms, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight = ctx.saved_tensors
        grad_input, grad_weight = npu_rms_norm_backward(grad_output, hidden_states, weight, inv_rms)
        return grad_input, grad_weight, None, None
