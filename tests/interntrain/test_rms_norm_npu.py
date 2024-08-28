# Copyright (c) 2024, DeepLink.

import torch
import torch_npu

from tests.core import call_module, allclose
from deeplink_ext.interntrain_ops.rms_norm import MixedFusedRMSNorm
from deeplink_ext.interntrain_ops.rms_norm_fallback import MixedRMSNormTorch


def test_MixedFusedRMSNorm():
    input_dtype_list = [torch.float32, torch.bfloat16, torch.float32, torch.float32]
    weight_dtype_list = [torch.float32, torch.bfloat16, torch.float16, torch.bfloat16]
    for input_dtype, weight_dtype in zip(input_dtype_list, weight_dtype_list):
        hidden_states_ext = torch.randn(
            1, 64, 32, 64, dtype=input_dtype, device="npu", requires_grad=True
        )
        hidden_states_ref = hidden_states_ext.cpu().detach().requires_grad_(True)
        # hidden_states_ext = hidden_states_ext.to('npu')

        output_ref, grad_ref = call_module(
            MixedRMSNormTorch(list(hidden_states_ref.shape)[-1], 1e-5).to(weight_dtype),
            hidden_states_ref,
        )

        # model = MixedFusedRMSNorm(list(hidden_states_ext.shape)[-1], 1e-5)
        # output_ext = model(hidden_states_ext)
        # output_ext.backward(torch.ones_like(hidden_states_ext).npu())
        # grad_ext = [hidden_states_ext.grad]

        output_ext, grad_ext = call_module(
            MixedFusedRMSNorm(list(hidden_states_ext.shape)[-1], 1e-5)
            .npu()
            .to(weight_dtype),
            hidden_states_ext,
        )
        print(f"grad_ref is None ? {grad_ref[0] is None}")
        print(f"grad_ext is None ? {grad_ext[0] is None}")

        assert allclose(
            output_ref, output_ext, rtol=1e-05, atol=1e-08
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the forward test!"
        assert allclose(
            grad_ref, grad_ext, rtol=1e-2, atol=1e-2
        ), f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the backward test!"


test_MixedFusedRMSNorm()
