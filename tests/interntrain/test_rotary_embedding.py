# Copyright (c) 2024, DeepLink.

import torch
from tests.core import call_autograd_func, allclose

from deeplink_ext.interntrain_ops.rotary_embedding import (
    ApplyRotaryEmb,
    ApplyRotaryEmbQKV_,
)
from deeplink_ext.interntrain_ops.rotary_embedding_fallback import (
    ApplyRotaryEmbTorch,
    ApplyRotaryEmbQKV_Torch,
)


def test_ApplyRotaryEmb():
    input_dtype_list = [torch.float16, torch.bfloat16, torch.float32]
    interleaved = False
    for input_dtype in input_dtype_list:
        input_ref = torch.randn(
            1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        input_ext = input_ref.clone().detach().requires_grad_()
        cos = torch.randn(64, 32, dtype=input_dtype, device="cuda")
        sin = torch.randn(64, 32, dtype=input_dtype, device="cuda")

        output_ref, grad_ref = call_autograd_func(
            ApplyRotaryEmbTorch, "cuda", input_dtype, input_ref, cos, sin, interleaved
        )
        output_ext, grad_ext = call_autograd_func(
            ApplyRotaryEmb, "cuda", input_dtype, input_ext, cos, sin, interleaved
        )
        assert allclose(
            output_ref, output_ext, rtol=1e-2, atol=5e-2
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the forward test!"
        assert allclose(
            grad_ref, grad_ext
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the backward test!"


def test_ApplyRotaryEmbQKV__qkv():
    # Note: For ascend, when dtype of input is fp32, the difference in calculation results is significant.
    input_dtype_list = [torch.float16, torch.bfloat16]
    interleaved = False
    for input_dtype in input_dtype_list:
        input_ref = torch.randn(
            1, 64, 3, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        input_ext = input_ref.clone().detach().requires_grad_()
        cos = torch.randn(64, 32, dtype=input_dtype, device="cuda")
        sin = torch.randn(64, 32, dtype=input_dtype, device="cuda")

        output_ref, grad_ref = call_autograd_func(
            ApplyRotaryEmbQKV_Torch,
            "cuda",
            input_dtype,
            input_ref,
            cos,
            sin,
            None,
            None,
            interleaved,
        )
        output_ext, grad_ext = call_autograd_func(
            ApplyRotaryEmbQKV_,
            "cuda",
            input_dtype,
            input_ext,
            cos,
            sin,
            None,
            None,
            interleaved,
        )

        assert allclose(
            output_ref, output_ext, rtol=1e-2, atol=5e-2
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmbQKV_ fails to pass the forward test!"

        assert allclose(
            grad_ref,
            grad_ext,
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmbQKV_ fails to pass the backward test!"
