# Copyright (c) 2024, DeepLink.

# TODO: add fallback for the newest internevo
# _not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."


# try:
#     from .rms_norm import RMSNorm, RMSNormWithNormalizedShape
# except:
#     print(
#         _not_impl.format(op_name="RMSNorm or RMSNormWithNormalizedShape"),
#     )
#     from .rms_norm_fallback import (
#         RMSNorm,
#         RMSNormWithNormalizedShape,
#     )


# try:
#     from .rotary_embedding import apply_rotary
# except:
#     print(_not_impl.format(op_name="apply_rotary"))
#     from .rotary_embedding_fallback import apply_rotary


# try:
#     from .mha import SelfAttention, CrossAttention
# except Exception as e:
#     print(_not_impl.format(op_name="mha"))
#     from .mha_fallback import SelfAttention, CrossAttention

__all__ = [
    "adamw_for_internlm",
    "DeepLinkSelfAttention",
    "DeepLinkCrossAttention",
    "DeepLinkMixedFusedRMSNorm",
    "DeeplinkApplyRotaryEmb",
    "DeeplinkApplyRotaryEmbQKV_",
]
