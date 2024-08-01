# Copyright (c) 2024, DeepLink.

_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .rotary_embedding import apply_rotary

except Exception as e:
    print(_not_impl.format(op_name="rotary_embedding"))
    print("Rotary Embedding currently does not support fallback!")

__all__ = [
    "apply_rotary",
]
