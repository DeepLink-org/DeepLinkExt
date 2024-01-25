# Copyright (c) 2024, DeepLink.

try:
    from .deeplink import DeepLinkApplyRotaryEmb, DeepLinkApplyRotaryEmbQKV_
except:
    print(
        "[deeplink_ext] rotary is not implemented in diopi. Falling back to the slower implementation.\n",
        end="",
    )
    from .fallback import (
        ApplyRotaryEmb as DeepLinkApplyRotaryEmb,
        ApplyRotaryEmbQKV_ as DeepLinkApplyRotaryEmbQKV_,
    )
from . import fallback
