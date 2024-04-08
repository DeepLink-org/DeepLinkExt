# Copyright (c) 2023, DeepLink.

import torch
from deeplink_ext.internevo_ops.rotary_embedding import apply_rotary
from deeplink_ext.internlm_ops.rotary_embeddinig_fallback import (
    apply_rotary as apply_rotary_fb,
)


def RotaryEmbTestFloat16() -> bool:
    input = torch.randn(1, 125, 16, 32, dtype=torch.float16).cuda()

    cos = torch.randn(217, 16, dtype=torch.float16).cuda()
    sin = torch.randn(217, 16, dtype=torch.float16).cuda()
    input1 = input.detach().clone()
    inplace = True
    interleaved = False

    res1 = apply_rotary_fb(input, cos, sin, interleaved=interleaved, inplace=inplace)
    res2 = apply_rotary(input1, cos, sin, interleaved=interleaved, inplace=inplace)

    # there is a little calculated error with ascend when dtype is float16
    return torch.allclose(res1, res2, atol=1e-2, rtol=1e-3)


def RotaryEmbTestFloat32() -> bool:
    input = torch.randn(1, 125, 16, 32, dtype=torch.float32).cuda()

    cos = torch.randn(217, 16, dtype=torch.float32).cuda()
    sin = torch.randn(217, 16, dtype=torch.float32).cuda()
    input1 = input.detach().clone()
    inplace = True
    interleaved = False

    res1 = apply_rotary_fb(input, cos, sin, interleaved=interleaved, inplace=inplace)
    res2 = apply_rotary(input1, cos, sin, interleaved=interleaved, inplace=inplace)

    return torch.allclose(res1, res2)


assert RotaryEmbTestFloat32()
assert RotaryEmbTestFloat16()
