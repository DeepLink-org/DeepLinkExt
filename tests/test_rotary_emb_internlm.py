# Copyright (c) 2023, DeepLink.

import torch
import deeplink_ext.internlm_ops.rotary as ext


def RotaryEmbTest() -> bool:
    input = torch.randn(1, 125, 16, 32, dtype=torch.float16).cuda()

    cos = torch.randn(217, 16, dtype=torch.float16).cuda()
    sin = torch.randn(217, 16, dtype=torch.float16).cuda()
    input1 = input.detach().clone()
    inplace = True
    interleaved = False

    res1 = ext.fallback.apply_rotary(
        input, cos, sin, interleaved=interleaved, inplace=inplace
    )
    res2 = ext.apply_rotary(input1, cos, sin, interleaved=interleaved, inplace=inplace)

    return torch.allclose(res1, res2)


assert RotaryEmbTest()
