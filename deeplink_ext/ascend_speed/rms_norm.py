import torch
import deeplink_ext.cpp_extensions as ext


assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        bias = torch.Tensor().cuda()
        output = torch.empty_like(hidden_states)
        input_dtype = hidden_states.dtype
        acc_dtype = (
            torch.float32
            if (input_dtype == torch.bfloat16 or input_dtype == torch.float16)
            else input_dtype
        )
        inv_rms = torch.empty_like(hidden_states, dtype=acc_dtype)
        ext.rms_norm(output, hidden_states, None, weight, bias, eps)
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
        return grad_input, grad_weight, None, None
