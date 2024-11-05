# Copyright (c) 2024, DeepLink.

from .easyllm_rms_norm_dipu import RMSNorm

__all__ = ["rms_norm"]


def rms_norm(x, weight, epsilon):
    return RMSNorm.apply(x, weight, epsilon)
