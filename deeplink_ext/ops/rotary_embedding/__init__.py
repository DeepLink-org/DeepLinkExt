_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

from .rotary_emb_utils import import_rotary_emb_funcs, patch_rotary_emb_funcs
ApplyRotaryEmb, ApplyRotaryEmbQKV_, apply_rotary = import_rotary_emb_funcs()
print(dir(apply_rotary))
patch_rotary_emb_funcs(apply_rotary)
