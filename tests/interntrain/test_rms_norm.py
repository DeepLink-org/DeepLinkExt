# Copyright (c) 2024, DeepLink.

import torch
from tests.core import call_module, allclose
from deeplink_ext.interntrain_ops.rms_norm import MixedFusedRMSNorm
from deeplink_ext.interntrain_ops.rms_norm_fallback import MixedRMSNormTorch


def test_MixedFusedRMSNorm():
    input_dtype_list = [torch.float16, torch.bfloat16, torch.float32, torch.float32]
    weight_dtype_list = [torch.float16, torch.bfloat16, torch.float16, torch.bfloat16]
    for input_dtype, weight_dtype in zip(input_dtype_list, weight_dtype_list):
        hidden_states_ref = torch.randn(
            1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        hidden_states_ext = hidden_states_ref.clone().detach().requires_grad_(True)

        output_ref, grad_ref = call_module(
            MixedRMSNormTorch(list(hidden_states_ref.shape)[-1], 1e-5)
            .cuda()
            .to(weight_dtype),
            hidden_states_ref,
        )
        output_ext, grad_ext = call_module(
            MixedFusedRMSNorm(list(hidden_states_ext.shape)[-1], 1e-5)
            .cuda()
            .to(weight_dtype),
            hidden_states_ext,
        )

        assert allclose(
            output_ref, output_ext, rtol=1e-05, atol=1e-08
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the forward test!"
        assert allclose(
            grad_ref, grad_ext, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the backward test!"
