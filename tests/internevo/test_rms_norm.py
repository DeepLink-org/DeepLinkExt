# Copyright (c) 2024, DeepLink.

import torch
from typing import Union, List
from tests.core import call_module, allclose
from deeplink_ext.internevo_ops.rms_norm import MixedFusedRMSNorm
from deeplink_ext.internevo_ops.rms_norm_fallback import MixedRMSNormTorch


def test_MixedFusedRMSNorm():
    input_dtype_list = [torch.float16, torch.bfloat16, torch.float32, torch.float32]
    weight_dtype_list = [torch.float16, torch.bfloat16, torch.float16, torch.bfloat16]
    for input_dtype, weight_dtype in zip(input_dtype_list, weight_dtype_list):
        hidden_states = torch.randn(
            1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        output_ref, grad_ref = call_module(
            MixedRMSNormTorch(list(hidden_states.shape)[-1], 1e-5)
            .cuda()
            .to(weight_dtype),
            hidden_states,
        )
        output, grad = call_module(
            MixedFusedRMSNorm(list(hidden_states.shape)[-1], 1e-5)
            .cuda()
            .to(weight_dtype),
            hidden_states,
        )
        assert allclose(
            output_ref, output, rtol=1e-05, atol=1e-08
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the forward test!"
        assert allclose(
            grad_ref, grad, rtol=1e-05, atol=1e-08
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the backward test!"
