# Copyright (c) 2024, DeepLink.

from typing import List
import torch
from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    import torch_npu
    # from torch_npu import npu_apply_adam_w as deeplink_ext_adamw
    deeplink_ext_adamw = torch.ops.npu.npu_apply_adam_w
elif platform_type == PlatformType.TORCH_DIPU:
    # import torch_dipu
    # assert torch_dipu.vendor_type == 'NPU', "ascend_speed framework only support NPU accelerators."
    from deeplink_ext.cpp_extensions import adamw as deeplink_ext_adamw
else:
    raise ImportError

__all__ = ["adamw"]


def adamw(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    norm_coeff_scale: float
):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """

    assert (
        maximize == False
    ), "The maximize parameter is not supported by diopiAdamW yet"

    for i, param in enumerate(params):
        if norm_coeff_scale is not None:
            grad = grads[i].float() * norm_coeff_scale
        else:
            grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if not max_exp_avg_sqs:
            max_exp_avg_sq = None
        else:
            max_exp_avg_sq = max_exp_avg_sqs[i]
        deeplink_ext_adamw(
            param,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            grad,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            amsgrad,
        )
    return params, exp_avgs, exp_avg_sqs
