# Copyright (c) 2024, DeepLink.

import torch
from torch.optim import AdamW
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

    # optimizer initialization
    adamw_optimizer = AdamW(
        params,
        lr_float,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=True,
        fused=False,
        found_inf=False,
    )
    # set the grad for params
    for i in range(len(params)):
        if grad_scale is not None:
            params[i].grad = grads[i].float() * grad_scale
        else:
            params[i].grad = grads[i]

    # set state_dict
    custom_state_dict = adamw_optimizer.state_dict()
    for i in range(len(params)):
        custom_state_dict["state"][i] = {
            "step": state_steps[0],
            "exp_avg": exp_avgs[i],
            "exp_avg_sq": exp_avg_sqs[i],
        }

    # load the custom_state_dict
    adamw_optimizer.load_state_dict(custom_state_dict)

    # perform a single optimization step
    adamw_optimizer.step()
    res_exp_avgs = []
    res_exp_avg_sqs = []
    for i in range(len(params)):
        res_exp_avgs.append(adamw_optimizer.state_dict()["state"][i]["exp_avg"])
        res_exp_avg_sqs.append(adamw_optimizer.state_dict()["state"][i]["exp_avg_sq"])
    return params, res_exp_avgs, res_exp_avg_sqs
