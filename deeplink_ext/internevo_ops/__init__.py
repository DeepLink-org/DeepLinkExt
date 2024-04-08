# Copyright (c) 2024, DeepLink.

# TODO: perfect the fallback of ext ops for the newest internevo
_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .adamw import fused_adamw
except Exception as e:
    print(_not_impl.format(op_name="adamw"))
    # TODO: add the fallback implementation of fused adamw op
    print(
        "The fallback implementation of the fused_adamw op is currently not supported!"
    )

try:
    from .flash_attention import FlashSelfAttention, FlashCrossAttention
except Exception as e:
    print(_not_impl.format(op_name="flash attention"))
    # TODO: add the fallback implementation of varlen flash attention op
    print(
        "The fallback implementation of the flash attention op currently only supports the padded mode!"
    )
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
    from .rotary_embedding_fallback import ApplyRotaryEmb, ApplyRotaryEmbQKV_


__all__ = [
    "fused_adamw",
    "FlashSelfAttention",
    "FlashCrossAttention",
    "MixedFusedRMSNorm",
    "ApplyRotaryEmb",
    "ApplyRotaryEmbQKV_",
]
