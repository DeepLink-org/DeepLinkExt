# Copyright (c) 2024, DeepLink.

import torch
from tests.core import call_func, allclose

from deeplink_ext.ascend_speed.rotary_embedding import RotaryEmbedding

from deeplink_ext.ascend_speed.rotary_embedding_fallback import RotaryEmbeddingTorch


def test_ApplyRotaryEmb():
    # Note: For ascend, when dtype of input is fp32, the difference in calculation results is significant.
    input_dtype_list = [torch.float16, torch.bfloat16]
    for input_dtype in input_dtype_list:
        input = torch.randn(
            4096, 1, 32, 128, dtype=input_dtype, device="cuda", requires_grad=True
        )
        cos = torch.randn(4096, 1, 1, 64, dtype=input_dtype, device="cuda")
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.randn(4096, 1, 1, 64, dtype=input_dtype, device="cuda")
        sin = torch.cat((sin, sin), dim=-1)

        output_ref, grad_ref = call_func(
            RotaryEmbeddingTorch, "cuda", input_dtype, input, cos, sin
        )
        output, grad = call_func(RotaryEmbedding, "cuda", input_dtype, input, cos, sin)
        assert allclose(
            output_ref, output, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the forward test!"
        assert allclose(
            grad_ref, grad
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the backward test!"
