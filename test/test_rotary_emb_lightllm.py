import torch
import torch_dipu
from einops import rearrange
import dipu_ext.ext_
from DipuExt_poc.ext_apply.lightllm.ext_apply_rotary import deeplink_rotary_emb

# lightllm的实现
def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0: dim // 2]
    x1 = x[:, :, dim // 2: dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)

# 构造示例输入数据
seq_len = 4
h = 2
dim = 4
# 创建输入张量 x，cos 和 sin
x = torch.randn(seq_len, h, dim).cuda()
cos = torch.randn(seq_len, dim // 2).cuda()
sin = torch.randn(seq_len, dim // 2).cuda()
x_copy = x.clone()
cos_copy = cos.clone()
sin_copy = sin.clone()
# DIOPI ext接口计算
output1 = deeplink_rotary_emb(x, cos, sin)
# lightllm 接口计算
output2 = torch_rotary_emb(x, cos, sin)
assert torch.allclose(output1, output2)
print("pass the lightllm rotary_emb test")
