# Copyright (c) 2024, DeepLink.

try:
    from .deeplink import DeepLinkRMSNorm, DeepLinkRMSNormWithNormalizedShape
except:
    print(
        "[deeplink_ext] rms_norm is not implemented in diopi. Falling back to the slower implementation.\n",
        end="",
    )
    from .fallback import RMSNorm as DeepLinkRMSNorm
from . import fallback

__all__ = ["DeepLinkRMSNorm", "DeepLinkRMSNormWithNormalizedShape", "fallback"]
