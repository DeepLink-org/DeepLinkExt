# Copyright (c) 2024, DeepLink.

import torch

from deeplink_ext.internevo_ops.rotary_embedding import (
    ApplyRotaryEmb,
    ApplyRotaryEmbQKV_,
)
from deeplink_ext.internevo_ops.rotary_embedding_fallback import (
    ApplyRotaryEmbTorch,
    ApplyRotaryEmbQKV_Torch,
)


def _run_rotary_embedding(
    rotary_embedding_module: type,
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
):
    class TestRotaryEmbedding(torch.nn.Module):
        def forward(self, input, cos, sin, interleaved):
            return rotary_embedding_module.apply(
                input,
                cos,
                sin,
                interleaved,
            )

    model = TestRotaryEmbedding()
    output = model(input, cos, sin, interleaved)
    output.backward(torch.ones_like(output))
    return output, input.grad


def test_multi_cases_for_rotary_embedding():
    input_dtype_list = [torch.float16, torch.bfloat16, torch.float32]
    interleaved = False
    for input_dtype in input_dtype_list:
        input = torch.rand(
            1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        cos = torch.rand(64, 32, dtype=input_dtype, device="cuda")
        sin = torch.rand(64, 32, dtype=input_dtype, device="cuda")

        output_ref, grad_ref = _run_rotary_embedding(
            ApplyRotaryEmbTorch, input, cos, sin, interleaved
        )
        output_ext, grad_ext = _run_rotary_embedding(
            ApplyRotaryEmb, input, cos, sin, interleaved
        )
        try:
            assert torch.allclose(output_ref, output_ext)
        except AssertionError:
            print(
                f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the forward test!"
            )
        else:
            print(
                f"When input dtype is {input_dtype}, ApplyRotaryEmb passes the forward test!"
            )

        try:
            assert torch.allclose(grad_ref, grad_ext)
        except AssertionError:
            print(
                f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the backward test!"
            )
        else:
            print(
                f"When input dtype is {input_dtype}, ApplyRotaryEmb passes the backward test!"
            )


test_multi_cases_for_rotary_embedding()
