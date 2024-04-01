import torch
import deeplink_ext.cpp_extensions as cpp_ext


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


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        bias = torch.Tensor().cuda()
        output, inv_rms = rms_norm(hidden_states, None, weight, bias, eps)
        ctx.save_for_backward(hidden_states, inv_rms, weight, bias)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = rms_norm_backward(
            hidden_states, grad_output, inv_rms, None, weight, bias, ctx.eps
        )
        return grad_input, grad_weight, None, None

