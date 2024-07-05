# Copyright (c) 2024, DeepLink.

from deeplink_ext.ascend_speed import RMSNorm

__all__ = ["rms_norm"]


def rms_norm(x, weight, epsilon):
    return RMSNorm.apply(x, weight, epsilon)
