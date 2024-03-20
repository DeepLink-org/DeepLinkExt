# Copyright (c) 2024, DeepLink.

try:
    from .deeplink import apply_rotary, RotaryEmbedding_AscendSpeed
except:
    print(
        "[deeplink_ext] rotary is not implemented in diopi. Falling back to the slower implementation.\n",
        end="",
    )
    from .fallback import apply_rotary
    RotaryEmbedding_AscendSpeed = None
from . import fallback

__all__ = ["apply_rotary", "fallback", "RotaryEmbedding_AscendSpeed"]
