# Copyright (c) 2024, DeepLink.

_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .adamw import AdamW
except Exception as e:
    print(_not_impl.format(op_name="adamw"))
    from torch.optim import AdamW

try:
    from .flash_attention import FlashSelfAttention, FlashCrossAttention
except Exception as e:
    print(_not_impl.format(op_name="flash attention"))
    from .flash_attention_fallback import SelfAttention as FlashSelfAttention
    from .flash_attention_fallback import CrossAttention as FlashCrossAttention


try:
    from .rms_norm import MixedFusedRMSNorm
except:
    print(
        _not_impl.format(op_name="RMSNorm"),
    )
    from .rms_norm_fallback import MixedRMSNormTorch as MixedFusedRMSNorm


try:
    from .rotary_embedding import ApplyRotaryEmb, ApplyRotaryEmbQKV_
except:
    print(_not_impl.format(op_name="rotary embedding"))
    from .rotary_embedding_fallback import ApplyRotaryEmbTorch as ApplyRotaryEmb
    from .rotary_embedding_fallback import ApplyRotaryEmbQKV_Torch as ApplyRotaryEmbQKV_


__all__ = [
    "AdamW",
    "FlashSelfAttention",
    "FlashCrossAttention",
    "MixedFusedRMSNorm",
    "ApplyRotaryEmb",
    "ApplyRotaryEmbQKV_",
]
