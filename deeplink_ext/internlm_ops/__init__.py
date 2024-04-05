# Copyright (c) 2024, DeepLink.

# TODO: add fallback for the newest internevo
_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."


try:
    from .flash_attention import DeepLinkSelfAttention, DeepLinkCrossAttention
except Exception as e:
    print(_not_impl.format(op_name="flash attention"))
    from .flash_attention_fallback import SelfAttention as DeepLinkSelfAttention
    from .flash_attention_fallback import CrossAttention as DeepLinkCrossAttention


try:
    from .rms_norm import DeepLinkMixedFusedRMSNorm
except:
    print(
        _not_impl.format(op_name="RMSNorm"),
    )
    from .rms_norm_fallback import MixedRMSNormTorch as DeepLinkMixedFusedRMSNorm


try:
    from .rotary_embedding import DeeplinkApplyRotaryEmb, DeeplinkApplyRotaryEmbQKV_
except:
    print(_not_impl.format(op_name="rotary embedding"))
    from .rotary_embedding_fallback import ApplyRotaryEmb as DeeplinkApplyRotaryEmb
    from .rotary_embedding_fallback import (
        ApplyRotaryEmbQKV_ as DeeplinkApplyRotaryEmbQKV_,
    )


__all__ = [
    "adamw_for_internlm",
    "DeepLinkSelfAttention",
    "DeepLinkCrossAttention",
    "DeepLinkMixedFusedRMSNorm",
    "DeeplinkApplyRotaryEmb",
    "DeeplinkApplyRotaryEmbQKV_",
]
