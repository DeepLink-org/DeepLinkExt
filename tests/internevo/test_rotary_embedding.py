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


class _TestRotaryEmbedding(torch.nn.Module):
    def __init__(self, rotary_embedding_module):
        super(_TestRotaryEmbedding, self).__init__()
        self.rotary_embedding_module = rotary_embedding_module

    def forward(self, input, cos, sin, interleaved):
        return self.rotary_embedding_module.apply(
            input,
            cos,
            sin,
            interleaved,
        )


class _TestRotaryEmbeddingQKV_(torch.nn.Module):
    def __init__(self, rotary_embedding_module):
        super(_TestRotaryEmbeddingQKV_, self).__init__()
        self.rotary_embedding_module = rotary_embedding_module

    def forward(self, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        return self.rotary_embedding_module.apply(
            qkv,
            cos,
            sin,
            cos_k,
            sin_k,
            interleaved,
        )


def _run_rotary_embedding(
    rotary_embedding_module: type,
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
):

    model = _TestRotaryEmbedding(rotary_embedding_module)
    output = model(input, cos, sin, interleaved)
    output.backward(torch.ones_like(output))
    return output, input.grad


def test_multi_cases_for_rotary_embedding():
    input_dtype_list = [torch.float16, torch.bfloat16, torch.float32]
    interleaved = False
    for input_dtype in input_dtype_list:
        input = torch.randn(
            1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        cos = torch.randn(64, 32, dtype=input_dtype, device="cuda")
        sin = torch.randn(64, 32, dtype=input_dtype, device="cuda")

        output_ref, grad_ref = _run_rotary_embedding(
            ApplyRotaryEmbTorch, input, cos, sin, interleaved
        )
        output_ext, grad_ext = _run_rotary_embedding(
            ApplyRotaryEmb, input, cos, sin, interleaved
        )

        assert torch.allclose(
            output_ref, output_ext, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the forward test!"

        assert torch.allclose(
            grad_ref, grad_ext
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the backward test!"
    print("Pass all forward and backward test cases of ApplyRotaryEmb successfully!")


def _run_rotary_embedding_qkv_(
    rotary_embedding_module: type,
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
):

    model = _TestRotaryEmbeddingQKV_(rotary_embedding_module)
    output = model(qkv, cos, sin, interleaved=interleaved)
    output.backward(torch.ones_like(output))
    return output, qkv.grad


def test_multi_cases_for_rotary_embedding_qkv_():
    input_dtype_list = [torch.float16, torch.bfloat16]
    interleaved = False
    for input_dtype in input_dtype_list:
        input_ref = torch.randn(
            1, 64, 3, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        input_ext = input_ref.clone().detach().cuda().requires_grad_()
        cos = torch.randn(64, 32, dtype=input_dtype, device="cuda")
        sin = torch.randn(64, 32, dtype=input_dtype, device="cuda")

        output_ref, grad_ref = _run_rotary_embedding_qkv_(
            ApplyRotaryEmbQKV_Torch, input_ref, cos, sin, interleaved
        )
        output_ext, grad_ext = _run_rotary_embedding_qkv_(
            ApplyRotaryEmbQKV_, input_ext, cos, sin, interleaved
        )

        assert torch.allclose(
            output_ref, output_ext, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the forward test!"

        assert torch.allclose(
            grad_ref, grad_ext
        ), f"When input dtype is {input_dtype}, ApplyRotaryEmb fails to pass the backward test!"
    print("Pass all forward and backward test cases of ApplyRotaryEmb successfully!")


test_multi_cases_for_rotary_embedding_qkv_()
