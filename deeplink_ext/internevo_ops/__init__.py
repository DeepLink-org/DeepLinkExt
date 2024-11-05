# Copyright (c) 2024, DeepLink.

try:
    from deeplink_ext.ops.adamw import AdamW
except Exception as e:
    print(_not_impl.format(op_name="adamw"))
    from torch.optim import AdamW

from deeplink_ext.ops.flash_attention import (
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_func,
)

from deeplink_ext.ops.rms_norm import MixedFusedRMSNorm

from deeplink_ext.ops.rotary_embedding import ApplyRotaryEmb

__all__ = [
    "AdamW",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_func",
    "MixedFusedRMSNorm",
    "ApplyRotaryEmb",
]
