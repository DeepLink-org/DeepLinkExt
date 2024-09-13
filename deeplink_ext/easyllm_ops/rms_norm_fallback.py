# Copyright (c) 2024, DeepLink.

import torch

__all__ = ["rms_norm_torch"]


def rms_norm_torch(x, weight, epsilon):
    input_dtype = x.dtype
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + epsilon)

    return (hidden_states * weight).to(input_dtype)
