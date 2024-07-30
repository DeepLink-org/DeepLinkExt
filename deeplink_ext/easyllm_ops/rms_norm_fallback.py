# Copyright (c) 2024, DeepLink.

import torch

__all__ = ["rms_norm_torch"]


def rms_norm_torch(x, weight, epsilon):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + epsilon)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(weight.dtype)

    return hidden_states * weight
