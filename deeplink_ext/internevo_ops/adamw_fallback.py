# Copyright (c) 2024, DeepLink.

import torch
from typing import List, Optional, Union


def fused_adamw_fallback(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    grad_scale: Union[float, torch.Tensor],
    found_inf: Optional[torch.Tensor],
):

    # some checks
    if found_inf is not None:
        raise RuntimeError("Deeplink Adamw with fused=True does not support found_inf")
    if maximize is not None and maximize is True:
        raise RuntimeError(
            "Deeplink Adamw with fused=True does not support maximize=True"
        )
    if amsgrad is not None and amsgrad is True:
        raise RuntimeError(
            "Deeplink Adamw with fused=True does not support amsgrad=True"
        )

    lr_float = float(lr.item()) if isinstance(lr, torch.Tensor) else lr

    for i in range(len(params)):
        if grad_scale is not None:
            grad = grads[i].float() * grad_scale
        else:
            grad = grads[i]

        if maximize:
            grad = -grad

        param = params[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        max_exp_avg_sq = None

        param -= lr_float * weight_decay * param
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
        exp_avg_bias_corrected = exp_avg / (1 - pow(beta1, state_steps[0] + 1))
        exp_avg_sq_bias_corrected = exp_avg_sq / (1 - pow(beta2, state_steps[0] + 1))

        if amsgrad:
            max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq_bias_corrected)
            param = param - lr_float * exp_avg_bias_corrected / (
                max_exp_avg_sq**0.5 + eps
            )
        else:
            param = param - lr_float * exp_avg_bias_corrected / (
                exp_avg_sq_bias_corrected**0.5 + eps
            )

        params[i] = param
        exp_avgs[i] = exp_avg
        exp_avg_sqs[i] = exp_avg_sq
    return params, exp_avgs, exp_avg_sqs
