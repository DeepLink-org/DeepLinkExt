import torch
import dipu_ext.ext_
from einops import rearrange


# Rotary_emb
# 本身就是基于pytorch的实现，所以不需要pytorch绕过代码
try:
    import dipu_ext.ext_
    print("USE ext apply_rotary")
    def deeplink_rotary_emb(x, cos, sin):
        seq_len, h, dim = x.shape
        cos = cos.view((seq_len, 1, dim // 2))
        sin = sin.view((seq_len, 1, dim // 2))
        output = torch.empty_like(x)
        dipu_ext.ext_.apply_rotary(output, x, cos, sin, False)
        return output
except:
    print("NOT USING ext apply_rotary")
    pass
