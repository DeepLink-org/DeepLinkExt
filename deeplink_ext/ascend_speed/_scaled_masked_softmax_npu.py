# Copyright (c) 2024, DeepLink.

import torch
import torch_npu

__all__ = ["ScaledMaskedSoftmax"]


class ScaledMaskedSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, scale, fixed_triu_mask):
        out = torch_npu.npu_scaled_masked_softmax(input, mask, scale, fixed_triu_mask)

        ctx.save_for_backward(out, mask)
        ctx.scale = scale
        ctx.fixed_triu_mask = fixed_triu_mask
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, mask = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)

        grad_input = torch_npu.npu_scaled_masked_softmax_backward(grad_output, out, mask, ctx.scale,
                                                                  ctx.fixed_triu_mask)

        return grad_input, None, None, None
