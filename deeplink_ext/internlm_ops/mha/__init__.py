# Copyright (c) 2023, DeepLink.

try:
    from .mha import DeepLinkSelfAttention, DeepLinkCrossAttention
except Exception as e:
    print(
        "[deeplink_ext] mha is not implemented in diopi. Falling back to the slower implementation."
    )
    from .fallback import (
        SelfAttention as DeepLinkSelfAttention,
        CrossAttention as DeepLinkCrossAttention,
    )
from . import fallback
