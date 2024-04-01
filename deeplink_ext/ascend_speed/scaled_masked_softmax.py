import torch
import deeplink_ext.cpp_extensions as ext


assert hasattr(ext, "scaled_masked_softmax_fwd") and hasattr(
    ext, "scaled_masked_softmax_bwd"
)


class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, scale, fixed_triu_mask):
        out = ext.scaled_masked_softmax_fwd(input, mask, scale, fixed_triu_mask)
        ctx.save_for_backward(out, mask)
        ctx.scale = scale
        ctx.fixed_triu_mask = fixed_triu_mask
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, mask = ctx.saved_tensors
        grad_input = ext.scaled_masked_softmax_bwd(
            grad_output, out, mask, ctx.scale, ctx.fixed_triu_mask
        )
        return grad_input, None, None, None
