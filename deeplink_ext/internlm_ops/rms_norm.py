# Copyright (c) 2024, DeepLink.

import numbers
import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")


__all__ = ["DeepLinkMixedFusedRMSNorm"]


class _DeepLinkMixedFusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps, normalized_shape):
        # ascend currently does not support dtype of input with higher precision than weight.
        higher_precision_dtype = torch.promote_types(hidden_states.dtype, weight.dtype)

        output = torch.empty_like(hidden_states)
        input_dtype = hidden_states.dtype
        acc_dtype = (
            torch.float32
            if input_dtype in [torch.bfloat16, torch.float16]
            else input_dtype
        )
        inv_rms = torch.empty(
            list(hidden_states.shape[:-1]) + [1],
            dtype=acc_dtype,
            device=hidden_states.device,
        )
        ext.rms_norm(
            output, inv_rms, hidden_states, normalized_shape, weight, None, eps
        )
        ctx.save_for_backward(hidden_states, inv_rms, weight)
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
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
            ctx.normalized_shape,
            ctx.eps,
        )
        return grad_input, grad_weight, None, None


class DeepLinkMixedFusedRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device="cuda"))
        self.variance_epsilon = eps
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, hidden_states):
        print("before: hidden_states.dtype", hidden_states.dtype)
        print("before: weight.dtype", self.weight.dtype)
        return _DeepLinkMixedFusedRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.normalized_shape,
        )
