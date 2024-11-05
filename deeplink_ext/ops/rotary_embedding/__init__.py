_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .internevo_rotary_embedding import ApplyRotaryEmb
except:
    print(_not_impl.format(op_name="rotary embedding"))
    from .internevo_rotary_embedding_fallback import (
        ApplyRotaryEmbTorch as ApplyRotaryEmb,
    )

try:
    from .interntrain_rotary_embedding import ApplyRotaryEmb, ApplyRotaryEmbQKV_
except:
    print(_not_impl.format(op_name="rotary embedding"))
    from .interntrain_rotary_embedding_fallback import (
        ApplyRotaryEmbTorch as ApplyRotaryEmb,
    )
    from .interntrain_rotary_embedding_fallback import (
        ApplyRotaryEmbQKV_Torch as ApplyRotaryEmbQKV_,
    )
