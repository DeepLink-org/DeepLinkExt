# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext


__all__ = ["RMSNorm", "RMSNormWithNormalizedShape"]


# 定义自定义的 autograd 函数
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
        return grad_input, grad_weight, grad_bias, None, None


class _DeepLinkRMSNormFunctionWithNormalizedShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps, normalized_shape):
        output, inv_rms = rms_norm(
            hidden_states.float(), normalized_shape, weight.float(), bias.float(), eps
        )
        output = output.half()
        inv_rms = inv_rms.half()
        ctx.save_for_backward(hidden_states, inv_rms, weight, bias, torch.tensor(eps))
        ctx.intermediate_results = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias, eps_tensor = ctx.saved_tensors
        eps = eps_tensor.item()
        normalized_shape = ctx.intermediate_results

        grad_input, grad_weight, grad_bias = rms_norm_backward(
            hidden_states.float(),
            grad_output.float(),
            inv_rms.float(),
            normalized_shape,
            weight.float(),
            bias.float(),
            eps,
        )
        return grad_input, grad_weight, grad_bias, None, None


# 定义一个 nn.Module 包裹这个自定义函数
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.zeros(hidden_size).cuda()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _DeepLinkRMSNormFunction.apply(
            hidden_states, self.weight, self.bias, self.variance_epsilon
        )


class RMSNormWithNormalizedShape(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.zeros(hidden_size).cuda()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _DeepLinkRMSNormFunctionWithNormalizedShape.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.variance_epsilon,
            self.weight.size(),
        )
