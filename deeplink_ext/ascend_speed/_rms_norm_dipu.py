# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")

__all__ = ["RMSNorm"]


class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        output = torch.empty_like(hidden_states)
        input_dtype = hidden_states.dtype
        acc_dtype = (
            torch.float32
            if input_dtype in [torch.bfloat16, torch.float16]
            else input_dtype
        )
        n = weight.dim()
        inv_rms = torch.empty(
            list(hidden_states.shape[:-n]),
            dtype=acc_dtype,
            device=hidden_states.device,
        )
        ext.rms_norm(output, inv_rms, hidden_states, weight.shape, weight, None, eps)
        ctx.save_for_backward(hidden_states, inv_rms, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight = ctx.saved_tensors
        grad_input = torch.empty_like(hidden_states)
        grad_weight = torch.empty_like(weight)
        ext.rms_norm_backward(
            grad_input,
            grad_weight,
            None,
            grad_output,
            hidden_states,
            weight,
            None,
            inv_rms,
            weight.shape,
            ctx.eps,
        )
        return grad_input, grad_weight, None, None
