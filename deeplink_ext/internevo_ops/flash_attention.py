# Copyright (c) 2024, DeepLink.

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    from ._flash_attention_npu import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_varlen_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
elif platform_type == PlatformType.TORCH_DIPU:
    from ._flash_attention_dipu import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_varlen_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
else:
    raise ImportError

__all__ = [
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
]
