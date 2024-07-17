# Copyright (c) 2024, DeepLink.

import torch
from tests.core import call_func, allclose
from deeplink_ext.internevo_ops.rotary_embedding import ApplyRotaryEmb
from deeplink_ext.internevo_ops.rotary_embedding_fallback import ApplyRotaryEmbTorch


def test_ApplyRotaryEmb():
    input_dtype_list = [torch.float16, torch.bfloat16]
    interleaved = False
    in_place_options = [False, True]
    for input_dtype in input_dtype_list:
        for in_place in in_place_options:
            input_ref = torch.randn(
                1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
            )
            input_ext = input_ref.clone().detach().requires_grad_()
            cos = torch.randn(64, 32, dtype=input_dtype, device="cuda")
            sin = torch.randn(64, 32, dtype=input_dtype, device="cuda")

            output_ref, grad_ref = call_func(
                ApplyRotaryEmbTorch,
                "cuda",
                input_dtype,
                input_ref,
                cos,
                sin,
                interleaved,
                in_place,
            )
            output_ext, grad_ext = call_func(
                ApplyRotaryEmb,
                "cuda",
                input_dtype,
                input_ext,
                cos,
                sin,
                interleaved,
                in_place,
            )
            assert allclose(
                output_ref, output_ext, rtol=1e-2, atol=5e-2
            ), f"When input dtype is {input_dtype} and in_place is {in_place}, ApplyRotaryEmb fails to pass the forward test!"
            assert allclose(
                grad_ref, grad_ext
            ), f"When input dtype is {input_dtype} and in_place is {in_place}, ApplyRotaryEmb fails to pass the backward test!"
