# Copyright (c) 2024, DeepLink.

_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

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

from deeplink_ext.ops.rms_norm import rms_norm
from deeplink_ext.ops.bert_padding import pad_input, unpad_input, index_first_axis

__all__ = [
    "AdamW",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_func",
    "rms_norm",
    "pad_input",
    "unpad_input",
    "index_first_axis",
]
