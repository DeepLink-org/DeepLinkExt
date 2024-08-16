# Copyright (c) 2024, DeepLink.
import numbers
import torch
import torch_npu
from torch.nn import init

from torch_npu import npu_rms_norm, npu_rms_norm_backward

__all__ = ["MixedFusedRMSNorm"]

# MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
# as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
# See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"


class _MixedFusedRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hs, weight, eps):
        out, rstd = npu_rms_norm(hs, weight, eps)
        ctx.save_for_backward(hs, weight, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (hs, weight, rstd) = ctx.saved_tensors
        grad_input, grad_weight = npu_rms_norm_backward(grad_output, hs, weight, rstd)
        return grad_input, grad_weight, None


class MixedFusedRMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        # TODO: Further optimization when there are device and dtype available.
        # factory_kwargs = {"device": device, "dtype": dtype}
        factory_kwargs = {}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, **factory_kwargs)).npu()
        self.eps = eps

    def forward(self, hidden_states):
        # out, _ = npu_rms_norm(hidden_states, self.weight, self.eps)
        # return out
        return _MixedFusedRMSNorm.apply(hidden_states, self.weight.to(hidden_states.dtype), self.eps)

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
