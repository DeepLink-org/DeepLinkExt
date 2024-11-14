_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from ._rotary_embedding import ApplyRotaryEmb, ApplyRotaryEmbQKV_
except:
    print(_not_impl.format(op_name="rotary embedding"))
    from ._rotary_embedding_fallback import (
        ApplyRotaryEmbTorch as ApplyRotaryEmb,
        ApplyRotaryEmbQKV_Torch as ApplyRotaryEmbQKV_,
    )
