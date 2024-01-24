# Copyright (c) 2024, DeepLink.

try:
    from .deeplink import DeepLinkApplyRotaryEmb, DeepLinkApplyRotaryEmbQKV_
except:
    print(
        "[deeplink_ext] rotary is not implemented in diopi. Falling back to the slower implementation."
    )
    from .fallback import (
        ApplyRotaryEmb as DeepLinkApplyRotaryEmb,
        ApplyRotaryEmbQKV_ as DeepLinkApplyRotaryEmbQKV_,
    )
from . import fallback
