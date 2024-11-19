_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

from .flash_attn_utils import import_flash_attn_modules, import_flash_attn_funcs

FlashSelfAttention, FlashCrossAttention = import_flash_attn_modules()
(
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_func,
) = import_flash_attn_funcs()

from .flash_attn_utils import patch_mha, patch_flash_attn_funcs

patch_mha(FlashSelfAttention, FlashCrossAttention)
patch_flash_attn_funcs(
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_func,
)
