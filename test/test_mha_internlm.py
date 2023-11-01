# Copyright (c) 2023, DeepLink.

import torch
import torch_dipu
import DipuExt_poc.ext_apply.internlm.ext_mha as ext_mha


def _run_self_attention(self_attn_module: type, qkv_data: torch.Tensor):
    qkv = qkv_data.clone().cuda().requires_grad_()
    self_attn = self_attn_module(causal=True).cuda()
    output = self_attn(qkv)
    output.backward(torch.ones_like(output))
    return output, qkv.grad


B = 2
S = 2
H = 2
D = 8
qkv = torch.randn(B, S, 3, H, D, dtype=torch.float16).cuda()
output_gold, grad_gold = _run_self_attention(ext_mha.fallback.SelfAttention, qkv)
output_ext, grad_ext = _run_self_attention(ext_mha.DeeplinkSelfAttention, qkv)
print(f"output_gold = {output_gold}")
print(f"output_ext = {output_ext}")
print(f"grad_gold = {grad_gold}")
print(f"grad_ext = {grad_ext}")
assert torch.allclose(output_gold, output_ext, atol=1e-3)
assert torch.allclose(grad_gold, grad_ext, atol=2e-3)
