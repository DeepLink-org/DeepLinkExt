# Copyright (c) 2024, DeepLink.

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    # from ._rotary_embedding_npu import ApplyRotaryEmb
    from .rotary_embedding_fallback import ApplyRotaryEmbTorch as ApplyRotaryEmb
elif platform_type == PlatformType.TORCH_DIPU:
    from ._rotary_embedding_dipu import ApplyRotaryEmb
else:
    raise ImportError

__all__ = ["ApplyRotaryEmb"]
