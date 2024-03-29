# Copyright (c) 2024, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "rms_norm")


# 定义自定义的 autograd 函数
class _DeepLinkRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps):
        output, inv_rms = ext.rms_norm(hidden_states, None, weight, bias, eps)

        ctx.save_for_backward(hidden_states, inv_rms, weight, bias, torch.tensor(eps))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias, eps_tensor = ctx.saved_tensors
        eps = eps_tensor.item()
        grad_input, grad_weight, grad_bias = ext.rms_norm_backward(
            hidden_states, grad_output, inv_rms, None, weight, bias, eps
        )
        return grad_input, grad_weight, grad_bias, None


class _DeepLinkRMSNormFunctionWithNormalizedShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, eps, normalized_shape):
        dtype = weight.dtype
        ctx.intermediate_results = normalized_shape
        fast = False # fp16 or bf16 
        if fast:
            if weight.dtype != dtype:
                weight = weight.to(dtype = dtype)
            if bias is not None and bias.dtype != dtype:
                bias = bias.to(dtype = dtype)
            output, inv_rms = ext.rms_norm(
                hidden_states, normalized_shape, weight, bias, eps
            )
            ctx.save_for_backward(hidden_states, inv_rms, weight, bias, torch.tensor(eps))
        else:
            hidden_states_float = hidden_states.float()
            weight_float = weight.float()
            bias_float = bias.float()
            output, inv_rms = ext.rms_norm(
                hidden_states_float, normalized_shape, weight_float, bias_float, eps
            )
            output = output.to(dtype = dtype)
            ctx.save_for_backward(hidden_states_float, inv_rms, weight_float, bias_float, torch.tensor(eps))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias, eps_tensor = ctx.saved_tensors
        eps = eps_tensor.item()
        normalized_shape = ctx.intermediate_results
        dtype = weight.dtype
        fast = False # fp16 or bf16 
        if fast:
            grad_input, grad_weight, grad_bias = ext.rms_norm_backward(
                hidden_states, grad_output, inv_rms, normalized_shape, weight, bias, eps
            )
        else:
            grad_input, grad_weight, grad_bias = ext.rms_norm_backward(
                hidden_states, grad_output.float(), inv_rms, normalized_shape, weight, bias, eps
            )
            grad_input = grad_input.to(dtype = dtype)
            # grad_weight = grad_weight.to(dtype = dtype)
            # if grad_bias is not None
            #     grad_bias = grad_bias.to(dtype = dtype)
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
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.zeros(normalized_shape).cuda()
        self.variance_epsilon = eps
        self.reset_parameters()

    def forward(self, hidden_states):
        return _DeepLinkRMSNormFunctionWithNormalizedShape.apply(
            hidden_states,
            self.weight,
            self.bias,
            self.variance_epsilon,
            self.weight.size(),
        )
    
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
