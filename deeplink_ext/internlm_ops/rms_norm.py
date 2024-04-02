# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as cpp_ext

__all__ = ["RMSNorm", "RMSNormWithNormalizedShape"]


def rms_norm_out(output, inv_rms, input, normalized_shape, weight, bias, eps):
    if None == normalized_shape:
        cpp_ext.rms_norm(output, inv_rms, input, weight.shape, weight, bias, eps)
    else:
        cpp_ext.rms_norm(output, inv_rms, input, normalized_shape, weight, bias, eps)


def rms_norm(input, normalized_shape, weight, bias, eps):
    output = torch.empty_like(input)
    inv_rms_shape = list(input.shape[:-1]) + [1]
    inv_rms = torch.empty(inv_rms_shape, dtype=input.dtype, device=input.device)
    rms_norm_out(output, inv_rms, input, normalized_shape, weight, bias, eps)

    return [output, inv_rms]


def rms_norm_backward_out(
    grad_input,
    grad_weight,
    grad_bias,
    grad_output,
    input,
    weight,
    bias,
    inv_rms,
    normalized_shape,
    eps,
):
    if None == normalized_shape:
        cpp_ext.rms_norm_backward(
            grad_input,
            grad_weight,
            grad_bias,
            grad_output,
            input,
            weight,
            bias,
            inv_rms,
            weight.shape,
            eps,
        )
    else:
        cpp_ext.rms_norm_backward(
            grad_input,
            grad_weight,
            grad_bias,
            grad_output,
            input,
            weight,
            bias,
            inv_rms,
            normalized_shape,
            eps,
        )


def rms_norm_backward(input, grad_output, inv_rms, normalized_shape, weight, bias, eps):
    grad_input = torch.empty_like(input)
    grad_weight = torch.empty_like(weight)
    if None == bias:
        grad_bias = None
    else:
        grad_bias = torch.empty_like(bias)
    rms_norm_backward_out(
        grad_input,
        grad_weight,
        grad_bias,
        grad_output,
        input,
        weight,
        bias,
        inv_rms,
        normalized_shape,
        eps,
    )

    return [grad_input, grad_weight, grad_bias]


# 定义自定义的 autograd 函数
class _DeepLinkRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps):
        output, inv_rms = rms_norm(hidden_states, None, weight, bias, eps)
        ctx.save_for_backward(hidden_states, inv_rms, weight, bias, torch.tensor(eps))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias, eps_tensor = ctx.saved_tensors
        eps = eps_tensor.item()
        grad_input, grad_weight, grad_bias = rms_norm_backward(
            hidden_states, grad_output, inv_rms, None, weight, bias, eps
        )
        return grad_input, grad_weight, grad_bias, None


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
