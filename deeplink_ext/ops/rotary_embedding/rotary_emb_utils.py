_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

def patch_rotary_emb_funcs(
    apply_rotary
):
    try:
        import rotary_emb
    except Exception as e:
        print("Unable to import rotary_emb, skip mocking flash_attn")
        return

    rotary_emb.apply_rotary = apply_rotary

def import_rotary_emb_funcs():
    try:
        from ._rotary_embedding_dipu import (
            ApplyRotaryEmb,
            ApplyRotaryEmbQKV_,
            apply_rotary,
    )
    except Exception as e:
        print(_not_impl.format(op_name="flash attention"))
        from ._rotary_embedding_fallback import (
            ApplyRotaryEmbTorch as ApplyRotaryEmb,
            ApplyRotaryEmbQKV_Torch as ApplyRotaryEmbQKV_,
            _torch_apply_rotary_func as apply_rotary,
        )
    return (
            ApplyRotaryEmb,
            ApplyRotaryEmbQKV_,
            apply_rotary,
    )
