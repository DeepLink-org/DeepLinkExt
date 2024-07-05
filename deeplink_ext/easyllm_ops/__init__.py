# Copyright (c) 2024, DeepLink.

_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .adamw import AdamW
except Exception as e:
    print(_not_impl.format(op_name="adamw"))
    from torch.optim import AdamW as AdamW

try:
    from .flash_attention import (
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_func,
    )
except Exception as e:
    print(_not_impl.format(op_name="flash attention"))
    print("Flash attention currently does not support fallback!")

try:
    from .rms_norm import rms_norm
except:
    print(
        _not_impl.format(op_name="RMSNorm"),
    )
    print("RMSNorm currently does not support fallback!")


__all__ = [
    "AdamW",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_func",
    "rms_norm",
]
