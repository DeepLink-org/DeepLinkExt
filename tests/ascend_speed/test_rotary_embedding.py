# Copyright (c) 2024, DeepLink.

import torch
from tests.core import call_autograd_func, allclose

from deeplink_ext.ascend_speed.rotary_embedding import RotaryEmbedding

from deeplink_ext.ascend_speed.rotary_embedding_fallback import RotaryEmbeddingTorch


def test_ApplyRotaryEmb():
    # Note: For ascend, when dtype of input is fp32, the difference in calculation results is significant.
    input_dtype_list = [torch.float16, torch.bfloat16]
    for input_dtype in input_dtype_list:
        input_ref = torch.randn(
            4096, 1, 32, 128, dtype=input_dtype, device="cuda", requires_grad=True
        )
        input_ext = input_ref.clone().detach().requires_grad_()
        cos = torch.randn(4096, 1, 1, 64, dtype=input_dtype, device="cuda")
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.randn(4096, 1, 1, 64, dtype=input_dtype, device="cuda")
        sin = torch.cat((sin, sin), dim=-1)

        output_ref, grad_ref = call_autograd_func(
            RotaryEmbeddingTorch, "cuda", input_dtype, input_ref, cos, sin
        )
        output_ext, grad_ext = call_autograd_func(
            RotaryEmbedding, "cuda", input_dtype, input_ext, cos, sin
        )
        assert allclose(
            output_ref, output_ext, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the forward test!"
        assert allclose(
            grad_ref, grad_ext
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the backward test!"
