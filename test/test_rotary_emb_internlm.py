import torch
import torch_dipu
from einops import rearrange
import dipu_ext.ext_
from ext_apply.internlm.ext_apply_rotary import TorchApplyRotaryEmbQKV_, DeeplLinkApplyRotaryEmbQKV_

torch_apply = TorchApplyRotaryEmbQKV_.apply
dipu_apply = DeeplLinkApplyRotaryEmbQKV_.apply
qkv = torch.randn(1, 125, 3, 16, 32, dtype=torch.float16).cuda()
cos = torch.randn(257,16, dtype=torch.float16).cuda()
sin =torch.randn(257,16, dtype=torch.float16).cuda()
qkv1 = qkv.clone()
cos1 = cos.clone()
sin1 = sin.clone()
cos_k = None
sin_k = None
interleaved = False
with torch.no_grad():
    res1 = torch_apply(qkv, cos, sin, cos_k, sin_k, interleaved)
    res2 = dipu_apply(qkv1, cos1, sin1, cos_k, sin_k, interleaved)
assert torch.allclose(res1,res2)
print("pass the internlm rotary_emb test")

