_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .internevo_flash_attention import (
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_func,
    )
except Exception as e:
    print(_not_impl.format(op_name="flash attention"))
    from .internevo_flash_attention_fallback import (
        flash_attn_qkvpacked_func_torch as flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func_torch as flash_attn_kvpacked_func,
        flash_attn_func_torch as flash_attn_func,
        flash_attn_varlen_qkvpacked_func_torch as flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_kvpacked_func_torch as flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_func_torch as flash_attn_varlen_func,
    )

try:
    from .interntrain_flash_attention import FlashSelfAttention, FlashCrossAttention
except Exception as e:
    print(_not_impl.format(op_name="flash attention"))
    from .interntrain_flash_attention_fallback import (
        SelfAttention as FlashSelfAttention,
    )
    from .interntrain_flash_attention_fallback import (
        CrossAttention as FlashCrossAttention,
    )
