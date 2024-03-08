# Copyright (c) 2024, DeepLink.

try:
    from .deeplink import DeepLinkApplyRotaryEmbQKV_, DeepLinkApplyRotaryEmb
except:
    print(
        "[deeplink_ext] rotary is not implemented in diopi. Falling back to the slower implementation.\n",
        end="",
    )
    from .fallback import (
        ApplyRotaryEmbQKV_ as DeepLinkApplyRotaryEmbQKV_,
        ApplyRotaryEmb as DeepLinkApplyRotaryEmb,
    )
from . import fallback

__all__ = ["DeepLinkApplyRotaryEmbQKV_", "DeepLinkApplyRotaryEmb", "fallback"]
