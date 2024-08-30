# Copyright (c) 2024, DeepLink.

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    from ._rms_norm_npu import RMSNorm
elif platform_type == PlatformType.TORCH_DIPU:
    from ._rms_norm_dipu import RMSNorm
else:
    raise ImportError

__all__ = ["RMSNorm"]
