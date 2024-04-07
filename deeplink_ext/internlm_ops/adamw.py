import torch
import deeplink_ext.cpp_extensions as ext
from typing import List, Optional, Union


assert hasattr(ext, "adamw")

__all__ = ["fused_adamw"]


def fused_adamw(
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
) -> None:
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
    lr_float = lr
    if isinstance(lr, torch.Tensor):
        lr_float = float(lr.item())
    for i, param in enumerate(params):
        if grad_scale is not None:
            grad = grads[i].float() * grad_scale
        else:
            grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if not max_exp_avg_sqs:
            max_exp_avg_sq = torch.Tensor.cuda()
        else:
            max_exp_avg_sq = max_exp_avg_sqs[i]
        ext.adamw(
            param,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            grad,
            lr_float,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            amsgrad,
        )
    return params, exp_avgs, exp_avg_sqs
