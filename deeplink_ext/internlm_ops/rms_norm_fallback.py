# Copyright (c) 2024, DeepLink.
# Copyright (c) 2022, NVIDIA CORPORATION.

import numbers

import torch
from torch.nn import init


__all__ = ["MixedRMSNormTorch"]


def manual_rms_norm(my_input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = my_input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    my_input = my_input * torch.rsqrt(variance + eps)

    if weight is None:
        return my_input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        my_input = my_input.to(weight.dtype)

    return weight * my_input


# adopted from https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm
# This torch implementation is equivalent to MixedFusedRMSNorm in apex.normalization.fused_layer_norm.
# MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
# as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
# See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"
class MixedRMSNormTorch(torch.nn.Module):
    """A custom PyTorch module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        # TODO: Further optimization when there are device and dtype available.
        # factory_kwargs = {"device": device, "dtype": dtype}
        factory_kwargs = {}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, **factory_kwargs))
        self.eps = eps
        self.reset_parameters()

    def forward(self, _input: torch.Tensor):
        return manual_rms_norm(_input, self.normalized_shape, self.weight, self.eps)

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
