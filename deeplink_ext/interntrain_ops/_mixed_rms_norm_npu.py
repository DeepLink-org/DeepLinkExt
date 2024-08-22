# Copyright (c) 2024, DeepLink.
import numbers
import torch
import torch_npu
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch_npu import npu_rms_norm

__all__ = ["MixedFusedRMSNorm"]


# MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
# as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
# See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"
def manual_rms_norm(my_input: Tensor, normalized_shape, weight: Tensor, eps, add_unit_offset=False):
    assert add_unit_offset == False
    assert len(normalized_shape) == 1

    input_dtype = my_input.dtype
    weight_dtype = weight.dtype

    acc_dtype = torch.promote_types(input_dtype, weight_dtype)
    out = npu_rms_norm(my_input.to(dtype=acc_dtype), weight.to(dtype=acc_dtype), eps)[0]
    if (out.dtype != weight_dtype):
        out = out.to(dtype=weight_dtype)
    return out


class MixedFusedRMSNorm(torch.nn.Module):
    """A custom PyTorch module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5, add_unit_offset=False):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape))
        self.add_unit_offset = add_unit_offset
        self.reset_parameters()

    def forward(self, _input: torch.Tensor):
        return manual_rms_norm(_input, self.normalized_shape, self.weight, self.eps, self.add_unit_offset)

    def reset_parameters(self):
        if self.add_unit_offset:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
