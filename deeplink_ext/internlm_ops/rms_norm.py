# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")


__all__ = ["RMSNorm", "RMSNormWithNormalizedShape"]


class _DeepLinkRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps):
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
        ext.rms_norm(output, inv_rms, hidden_states, weight.shape, weight, bias, eps)
        ctx.save_for_backward(hidden_states, inv_rms, weight, bias)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias = ctx.saved_tensors
        grad_input = torch.empty_like(hidden_states)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(bias)
        ext.rms_norm_backward(
            grad_input,
            grad_weight,
            grad_bias,
            grad_output,
            hidden_states,
            weight,
            bias,
            inv_rms,
            weight.shape,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None


class _DeepLinkRMSNormFunctionWithNormalizedShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps, normalized_shape):
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
            output, inv_rms, hidden_states, normalized_shape, weight, bias, eps
        )
        ctx.save_for_backward(hidden_states, inv_rms, weight, bias)
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias = ctx.saved_tensors
        grad_input = torch.empty_like(hidden_states)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(bias)
        ext.rms_norm_backward(
            grad_input,
            grad_weight,
            grad_bias,
            grad_output,
            hidden_states,
            weight,
            bias,
            inv_rms,
            ctx.normalized_shape,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None, None


class DeepLinkRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device="cuda"))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape, device="cuda"))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _DeepLinkRMSNormFunction.apply(
            hidden_states, self.weight, self.bias, self.variance_epsilon
        )


class DeepLinkRMSNormWithNormalizedShape(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device="cuda"))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape, device="cuda"))
        self.variance_epsilon = eps
        self.normalized_shape = (
            torch.Size(normalized_shape)
            if hasattr(normalized_shape, "__iter__")
            else torch.Size((normalized_shape,))
        )

    def forward(self, hidden_states):
        return _DeepLinkRMSNormFunctionWithNormalizedShape.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.variance_epsilon,
            self.normalized_shape,
        )
