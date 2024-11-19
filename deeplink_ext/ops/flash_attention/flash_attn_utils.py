_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."


def patch_mha(CustomFlashSelfAttention, CustomFlashCrossAttention):
    FlashSelfAttention, FlashCrossAttention = (
        CustomFlashSelfAttention,
        CustomFlashCrossAttention,
    )
    try:
        import flash_attn.modules.mha as mha
    except Exception as e:
        print("Unable to import flash_attn, skip mocking flash_attn")
        return

    mha.FlashSelfAttention = FlashSelfAttention
    mha.FlashCrossAttention = FlashCrossAttention


def import_flash_attn_modules():
    try:
        from .interntrain_flash_attention import FlashSelfAttention
        from .interntrain_flash_attention import FlashCrossAttention
    except Exception as e:
        print(_not_impl.format(op_name="flash attention"))
        from .interntrain_flash_attention import SelfAttention as FlashSelfAttention
        from .interntrain_flash_attention import CrossAttention as FlashCrossAttention

    return (FlashSelfAttention, FlashCrossAttention)


def patch_flash_attn_funcs(
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_func,
):
    try:
        import flash_attn
    except Exception as e:
        print("Unable to import flash_attn, skip mocking flash_attn")
        return

    flash_attn.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    flash_attn.flash_attn_kvpacked_func = flash_attn_kvpacked_func
    flash_attn.flash_attn_func = flash_attn_func
    flash_attn.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    flash_attn.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    flash_attn.flash_attn_varlen_func = flash_attn_varlen_func


def import_flash_attn_funcs():
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
    return (
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_func,
    )
