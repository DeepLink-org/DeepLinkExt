# Copyright (c) 2024, DeepLink.

import torch
from tests.core import calculate_fwd_and_bwd, allclose
from deeplink_ext.easyllm_ops.rms_norm import rms_norm
from deeplink_ext.easyllm_ops.rms_norm_fallback import rms_norm_torch


def test_rms_norm():
    input_dtype_list = [torch.float16, torch.bfloat16]
    weight_dtype_list = [torch.float16, torch.bfloat16]
    for input_dtype, weight_dtype in zip(input_dtype_list, weight_dtype_list):
        hidden_states_ref = torch.randn(
            1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True
        )
        hidden_states_ext = hidden_states_ref.clone().detach().requires_grad_(True)

        weight_ref = torch.nn.Parameter(
            torch.ones(
                list(hidden_states_ref.shape)[-1], dtype=weight_dtype, device="cuda"
            ),
            requires_grad=True,
        )
        weight_ext = weight_ref.clone().detach().requires_grad_(True)

        epsilon = 1e-5

        output_ref, grad_ref = calculate_fwd_and_bwd(
            rms_norm_torch,
            hidden_states_ref,
            weight_ref,
            epsilon,
        )

        output_ext, grad_ext = calculate_fwd_and_bwd(
            rms_norm,
            hidden_states_ext,
            weight_ext,
            epsilon,
        )

        assert allclose(
            output_ref, output_ext, rtol=1e-05, atol=1e-5
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, RMSNorm fails to pass the forward test!"
        assert allclose(
            grad_ref, grad_ext, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, RMSNorm fails to pass the backward test!"
