# Copyright (c) 2024, DeepLink.

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    from ._flash_attention_npu import FlashSelfAttention, FlashCrossAttention
elif platform_type == PlatformType.TORCH_DIPU:
    from ._flash_attention_dipu import FlashSelfAttention, FlashCrossAttention
else:
    raise ImportError

__all__ = ["FlashSelfAttention", "FlashCrossAttention"]
