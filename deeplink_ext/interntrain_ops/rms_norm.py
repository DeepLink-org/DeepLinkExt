# Copyright (c) 2024, DeepLink.

import torch

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type
platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    from ._mixed_rms_norm_npu import MixedFusedRMSNorm
elif platform_type == PlatformType.TORCH_DIPU:
    from ._mixed_rms_norm_dipu import MixedFusedRMSNorm
else:
    raise ImportError

__all__ = ["MixedFusedRMSNorm"]
