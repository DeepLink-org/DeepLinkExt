# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext


__all__ = ["RMSNorm", "RMSNormWithNormalizedShape"]

assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps):
        output = torch.empty_like(hidden_states)
        input_dtype = hidden_states.dtype
        acc_dtype = (
            torch.float32
            if input_dtype in [torch.bfloat16, torch.float16]
            else input_dtype
        )
        inv_rms = torch.empty_like(hidden_states, dtype=acc_dtype)
        ext.rms_norm(output, inv_rms, hidden_states, None, weight, bias, eps)
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
            hidden_states,
            grad_output,
            inv_rms,
            None,
            weight,
            bias,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None


class _RMSNormFunctionWithNormalizedShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps, normalized_shape):
        output = torch.empty_like(hidden_states)
        input_dtype = hidden_states.dtype
        acc_dtype = (
            torch.float32
            if input_dtype in [torch.bfloat16, torch.float16]
            else input_dtype
        )
        inv_rms = torch.empty_like(hidden_states, dtype=acc_dtype)
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
            hidden_states,
            grad_output,
            inv_rms,
            ctx.normalized_shape,
            weight,
            bias,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None, None


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.zeros(hidden_size).cuda()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _RMSNormFunction.apply(
            hidden_states, self.weight, self.bias, self.variance_epsilon
        )


class RMSNormWithNormalizedShape(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.zeros(hidden_size).cuda()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _RMSNormFunctionWithNormalizedShape.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.variance_epsilon,
            self.weight.size(),
        )
