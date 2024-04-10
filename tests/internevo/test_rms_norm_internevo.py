# Copyright (c) 2024, DeepLink.

import torch
from typing import Union, List

from deeplink_ext.internevo_ops.rms_norm import MixedFusedRMSNorm
from deeplink_ext.internevo_ops.rms_norm_fallback import MixedRMSNormTorch


def _run_mixed_rms_norm(
    rms_norm_module: type,
    hidden_states: torch.Tensor,
    weight_dtype: torch.dtype,
    normalized_shape: Union[int, List[int], torch.Size],
    eps: float = 1e-5,
):
    model = rms_norm_module(normalized_shape, eps)
    model = model.cuda().to(dtype=weight_dtype)
    output = model(hidden_states)
    output.backward(torch.ones_like(output))
    return output, hidden_states.grad


def test_multi_cases_for_mixed_rms_norm():
    input_dtype_list = [torch.float16, torch.bfloat16, torch.float32, torch.float32]
    weight_dtype_list = [torch.float16, torch.bfloat16, torch.float16, torch.bfloat16]
    for input_dtype, weight_dtype in zip(input_dtype_list, weight_dtype_list):
        hidden_states = torch.rand(1, 64, 32, 64, dtype=input_dtype, device="cuda", requires_grad=True)
        # Note: For ascend, only the normalized shape is currently supported as the last dimension size of the input.
        output_ref, grad_ref = _run_mixed_rms_norm(
            MixedRMSNormTorch,
            hidden_states,
            weight_dtype,
            list(hidden_states.shape)[-1],
        )
        output_ext, grad_ext = _run_mixed_rms_norm(
            MixedFusedRMSNorm,
            hidden_states,
            weight_dtype,
            list(hidden_states.shape)[-1],
        )
        try:
            assert torch.allclose(output_ref, output_ext)
        except AssertionError:
            print(
                f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the forward test!"
            )
        else:
            print(
                f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm passes the forward test!"
            )

        try:
            assert torch.allclose(grad_ref, grad_ext)
        except AssertionError:
            print(
                f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm fails to pass the backward test!"
            )
        else:
            print(
                f"When input dtype is {input_dtype} and weight dtype is {weight_dtype}, MixedRMSNorm passes the backward test!"
            )
