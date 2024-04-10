# Copyright (c) 2024, DeepLink.

import numbers
import torch
from torch.nn import init

import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")


__all__ = ["MixedFusedRMSNorm"]


# MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
# as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
# See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"
class _MixedFusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps, normalized_shape):
        # ascend currently does not support dtype of hidden_states with higher precision than weight.
        # record original dtype of hidden_states and weight
        input_dtype = hidden_states.dtype
        weight_dtype = weight.dtype

        acc_dtype = torch.float32 if input_dtype in [torch.bfloat16, torch.float16] else input_dtype
        inv_rms = torch.empty(
            list(hidden_states.shape[:-1]) + [1],
            dtype=acc_dtype,
            device=hidden_states.device,
        )

        higher_precision = torch.promote_types(input_dtype, weight_dtype)
        output_higher_precision = torch.empty_like(hidden_states, dtype=higher_precision)
        hidden_states_higher_precision = hidden_states.to(dtype=higher_precision)
        weight_higher_precision = weight.to(dtype=higher_precision)

        ext.rms_norm(
            output_higher_precision,
            inv_rms,
            hidden_states_higher_precision,
            normalized_shape,
            weight_higher_precision,
            None,
            eps,
        )

        ctx.save_for_backward(hidden_states_higher_precision, inv_rms, weight_higher_precision)
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        ctx.input_dtype = input_dtype
        ctx.weight_dtype = weight_dtype

        return output_higher_precision.to(dtype=weight_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (
            hidden_states_higher_precision,
            inv_rms,
            weight_higher_precision,
        ) = ctx.saved_tensors

        grad_input_higher_precision = torch.empty_like(hidden_states_higher_precision)
        grad_weight_higher_precision = torch.empty_like(weight_higher_precision)

        ext.rms_norm_backward(
            grad_input_higher_precision,
            grad_weight_higher_precision,
            None,
            grad_output.to(dtype=grad_input_higher_precision.dtype),
            hidden_states_higher_precision,
            weight_higher_precision,
            None,
            inv_rms,
            ctx.normalized_shape,
            ctx.eps,
        )

        return (
            grad_input_higher_precision.to(dtype=ctx.input_dtype),
            grad_weight_higher_precision.to(dtype=ctx.weight_dtype),
            None,
            None,
        )


class MixedFusedRMSNorm(torch.nn.Module):
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

    def forward(self, hidden_states):
        return _MixedFusedRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.eps,
            self.normalized_shape,
        )

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
