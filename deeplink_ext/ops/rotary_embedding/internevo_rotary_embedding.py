# Copyright (c) 2024, DeepLink.

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_DIPU:
    from .internevo_rotary_embedding_dipu import ApplyRotaryEmb
else:
    raise ImportError

__all__ = ["ApplyRotaryEmb"]
