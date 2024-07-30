# Copyright (c) 2024, DeepLink.

_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .adamw import AdamW
except Exception as e:
    print(_not_impl.format(op_name="adamw"))
    from torch.optim import AdamW

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
    from .flash_attention_fallback import (
        flash_attn_qkvpacked_func_torch as flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func_torch as flash_attn_kvpacked_func,
        flash_attn_func_torch as flash_attn_func,
        flash_attn_varlen_qkvpacked_func_torch as flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_kvpacked_func_torch as flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_func_torch as flash_attn_varlen_func,
    )

try:
    from .rms_norm import rms_norm
except:
    print(
        _not_impl.format(op_name="RMSNorm"),
    )
    from .rms_norm_fallback import rms_norm_torch as rms_norm

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
