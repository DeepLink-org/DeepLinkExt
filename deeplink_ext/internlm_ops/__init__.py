# Copyright (c) 2024, DeepLink.

from . import mha


_not_impl = "[deeplink_ext] %s is not implemented in diopi. Falling back to the slower torch implementation."


try:
    from .rms_norm import RMSNorm, RMSNormWithNormalizedShape
except:
    print(
        _not_impl.format("RMSNorm or RMSNormWithNormalizedShape"),
    )
    from .rms_norm_fallback import (
        RMSNorm as RMSNorm,
        RMSNorm as RMSNormWithNormalizedShape,
    )


try:
    from .rotary_embedding import apply_rotary
except:
    print( _not_impl.format("apply_rotary"))
    from .rotary_embeddinig_fallback import apply_rotary



__all__ = ["mha", "RMSNorm", "RMSNormWithNormalizedShape", "apply_rotary"]