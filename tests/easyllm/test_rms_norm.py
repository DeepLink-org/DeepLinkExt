# Copyright (c) 2024, DeepLink.

import numbers

import torch
from tests.core import call_module, allclose, call_func
from deeplink_ext.easyllm_ops.rms_norm import RMSNorm


class RMSNormTorch(torch.nn.Module):
    """A custom PyTorch module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def _norm(self, input, dims):
        return input * torch.rsqrt(input.pow(2).mean(dims, keepdim=True) + self.eps)

    def forward(self, input: torch.Tensor):
        dims = tuple(i for i in range(-1, -len(self.normalized_shape) - 1, -1))
        return self._norm(input.float(), dims).type_as(input) * self.weight


def test_RMSNorm():
    input_dtype_list = [torch.float16, torch.bfloat16]
    weight_dtype_list = [torch.float16, torch.bfloat16]
    for input_dtype, weight_dtype in zip(input_dtype_list, weight_dtype_list):
        hidden_states_ref = torch.randn(
            1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        hidden_states_ext = hidden_states_ref.clone().detach().requires_grad_(True)

        weight = torch.nn.Parameter(
            torch.ones(
                list(hidden_states_ext.shape)[-1], dtype=weight_dtype, device="cuda"
            ),
            requires_grad=False,
        )

        output_ref, grad_ref = call_module(
            RMSNormTorch(list(hidden_states_ref.shape)[-1], 1e-5)
            .cuda()
            .to(weight_dtype),
            hidden_states_ref,
        )

        output_ext, grad_ext = call_func(
            RMSNorm, "cuda", weight_dtype, hidden_states_ext, weight, 1e-5
        )

        assert allclose(
            output_ref, output_ext, rtol=1e-05, atol=1e-5
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, RMSNorm fails to pass the forward test!"
        assert allclose(
            grad_ref, grad_ext, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, RMSNorm fails to pass the backward test!"
