# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "rms_norm")


# 定义自定义的 autograd 函数
class _DeepLinkRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps):
        output = torch.empty_like(hidden_states)
        inv_rms_shape = list(hidden_states.shape[:-1], 1)
        inv_rms = torch.empty(
            inv_rms_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        ext.rms_norm(output, inv_rms, hidden_states, None, weight, bias, eps)

        ctx.save_for_backward(hidden_states, inv_rms, weight, bias, torch.tensor(eps))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias, eps_tensor = ctx.saved_tensors
        eps = eps_tensor.item()

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
            None,
            eps
        )
        return grad_input, grad_weight, grad_bias, None


class _DeepLinkRMSNormFunctionWithNormalizedShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps, normalized_shape):
        output = torch.empty_like(hidden_states, dtype=torch.float32)
        inv_rms_shape = list(hidden_states.shape[:-1], 1)
        inv_rms = torch.empty(
            inv_rms_shape, dtype=torch.float32, device=hidden_states.device
        )
        ext.rms_norm(
            output,
            inv_rms,
            hidden_states.float(),
            normalized_shape,
            weight.float(),
            bias.float(),
            eps
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

        grad_input = torch.empty_like(hidden_states, dtype=torch.float32)
        grad_weight = torch.empty_like(weight, dtype=torch.float32)
        grad_bias = torch.empty_like(bias, dtype=torch.float32)
        ext.rms_norm_backward(
            grad_input,
            grad_weight,
            grad_bias,
            grad_output.float(),
            hidden_states.float(),
            weight.float(),
            bias.float(),
            inv_rms.float(),
            normalized_shape,
            eps
        )
        grad_output = grad_output.half()
        hidden_states = hidden_states.half()
        inv_rms = inv_rms.half()
        return grad_input, grad_weight, grad_bias, None, None


# 定义一个 nn.Module 包裹这个自定义函数
class DeepLinkRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.zeros(hidden_size).cuda()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _DeepLinkRMSNormFunction.apply(
            hidden_states, self.weight, self.bias, self.variance_epsilon
        )


class DeepLinkRMSNormWithNormalizedShape(torch.nn.Module):
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
