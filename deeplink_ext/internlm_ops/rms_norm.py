# Copyright (c) 2024, DeepLink.

import numbers
import torch
from torch.nn import init

import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "rms_norm") and hasattr(ext, "rms_norm_backward")


__all__ = ["DeepLinkMixedFusedRMSNorm"]


class _DeepLinkMixedFusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps, normalized_shape):
        # ascend currently does not support dtype of input with higher precision than weight.
        higher_precision_dtype = torch.promote_types(hidden_states.dtype, weight.dtype)

        output = torch.empty_like(hidden_states)
        input_dtype = hidden_states.dtype
        acc_dtype = (
            torch.float32
            if input_dtype in [torch.bfloat16, torch.float16]
            else input_dtype
        )
        inv_rms = torch.empty(
            list(hidden_states.shape[:-1]) + [1],
            dtype=acc_dtype,
            device=hidden_states.device,
        )
        ext.rms_norm(
            output, inv_rms, hidden_states, normalized_shape, weight, None, eps
        )
        ctx.save_for_backward(hidden_states, inv_rms, weight)
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, inv_rms, weight = ctx.saved_tensors
        grad_input = torch.empty_like(hidden_states)
        grad_weight = torch.empty_like(weight)
        ext.rms_norm_backward(
            grad_input,
            grad_weight,
            None,
            grad_output,
            hidden_states,
            weight,
            None,
            inv_rms,
            ctx.normalized_shape,
            ctx.eps,
        )
        return grad_input, grad_weight, None, None


class DeepLinkMixedFusedRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = torch.nn.Parameter(
            torch.ones(normalized_shape, device=self._device)
        )

    def forward(self, hidden_states):
        return _DeepLinkMixedFusedRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.eps,
            self.normalized_shape,
        )

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
