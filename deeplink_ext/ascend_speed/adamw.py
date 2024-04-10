# Copyright (c) 2024, DeepLink.

from typing import List
import torch
import deeplink_ext.cpp_extensions as ext


assert hasattr(ext, "adamw")

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

    assert maximize == False, "The maximize parameter is not supported by diopiAdamW yet"

    for i, param in enumerate(params):
        if norm_coeff_scale is not None:
            grad = grads[i].float() * norm_coeff_scale
        else:
            grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if not max_exp_avg_sqs:
            max_exp_avg_sq = torch.Tensor().cuda()
        else:
            max_exp_avg_sq = max_exp_avg_sqs[i]
        ext.adamw(
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
