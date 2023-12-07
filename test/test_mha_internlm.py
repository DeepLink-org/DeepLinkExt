# Copyright (c) 2023, DeepLink.

import torch
import torch_dipu
import DeepLinkExt.ext_apply.internlm.ext_mha as ext_mha


def _run_self_attention(self_attn_module: type, qkv_data: torch.Tensor):
    qkv = qkv_data.clone().cuda().requires_grad_()
    self_attn = self_attn_module(causal=True).cuda()
    output = self_attn(qkv)
    output.backward(torch.ones_like(output))
    return output, qkv.grad


def _run_cross_attention(
    cross_attn_module: type, q_data: torch.Tensor, kv_data: torch.Tensor
):
    q = q_data.clone().cuda().requires_grad_()
    kv = kv_data.clone().cuda().requires_grad_()
    self_attn = cross_attn_module(causal=True).cuda()
    output = self_attn(q, kv)
    output.backward(torch.ones_like(output))
    return output, q.grad, kv.grad


B = 2
S = 2
H = 2
D = 8
qkv = torch.randn(B, S, 3, H, D, dtype=torch.float16).cuda()
output_gold, grad_gold = _run_self_attention(ext_mha.fallback.SelfAttention, qkv)
output_ext, grad_ext = _run_self_attention(ext_mha.DeepLinkSelfAttention, qkv)
assert torch.allclose(output_gold, output_ext, atol=1e-3)
print("SelfAttention forward test pass")
assert torch.allclose(grad_gold, grad_ext, atol=2e-3)
print("SelfAttention backward test pass")

q = qkv[:, :, 0]
kv = qkv[:, :, 1:]
output_gold, dq_gold, dkv_gold = _run_cross_attention(
    ext_mha.fallback.CrossAttention, q, kv
)
output_ext, dq_ext, dkv_ext = _run_cross_attention(
    ext_mha.DeepLinkCrossAttention, q, kv
)
assert torch.allclose(output_gold, output_ext, atol=1e-3)
print("CrossAttention forward test pass")
assert torch.allclose(dq_gold, dq_ext, atol=2e-3)
assert torch.allclose(dkv_gold, dkv_ext, atol=2e-3)
print("CrossAttention backward test pass")
