# Copyright (c) 2024, DeepLink.
import warnings
from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type


def _init():
    # deeplink_ext is developed based on dipu
    # so we explicitly import torch_dipu to guarantees that torch is patched by dipu
    platform_type = deeplink_ext_get_platform_type()
    if platform_type == PlatformType.TORCH_DIPU:
        import torch_dipu
    elif platform_type == PlatformType.TORCH_NPU:
        warnings.warn("DeepLinkExt using torch_npu ...", ImportWarning)
        import torch_npu
    else:
        raise ImportError


_init()
