# Copyright (c) 2024, DeepLink.

_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from deeplink_ext.ops.adamw import AdamW
except Exception as e:
    print(_not_impl.format(op_name="adamw"))
    from torch.optim import AdamW

from deeplink_ext.ops.flash_attention import FlashSelfAttention, FlashCrossAttention
from deeplink_ext.ops.rms_norm import MixedFusedRMSNorm
from deeplink_ext.ops.rotary_embedding import ApplyRotaryEmb, ApplyRotaryEmbQKV_, apply_rotary


__all__ = [
    "AdamW",
    "FlashSelfAttention",
    "FlashCrossAttention",
    "MixedFusedRMSNorm",
    "ApplyRotaryEmb",
    "ApplyRotaryEmbQKV_",
    "apply_rotary",
]
