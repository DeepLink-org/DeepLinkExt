# Copyright (c) 2023, DeepLink.

import torch
from deeplink_ext.internlm_ops.rms_norm.deeplink import rms_norm, rms_norm_backward

# 定义输入张量
input = torch.randn(5, 5, requires_grad=True).cuda()

# 定义权重和偏差张量
weight = torch.randn(5, requires_grad=True).cuda()
bias = torch.randn(5, requires_grad=True).cuda()

# 定义输出梯度张量
grad_output = torch.randn(5, 5).cuda()

# 归一化的形状通常是输入张量的形状
normalized_shape = torch.tensor([5, 5], dtype=torch.long).cuda()

print(input.is_dipu)
output, inv_rms = rms_norm(input, None, weight, bias, 1e-6)

grad_input, grad_weight, grad_bias = rms_norm_backward(
    input, grad_output, inv_rms, None, weight, bias, 1e-6
)

print("Output:", output)
print("Grad Input:", grad_input)
print("Grad Weight:", grad_weight)
print("Grad Bias:", grad_bias)

input.requires_grad_(True)
weight.requires_grad_(True)
bias.requires_grad_(True)
b = input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
grads = torch.autograd.grad(b, [input, weight, bias], grad_output, allow_unused=True)
assert torch.allclose(output, b)
assert torch.allclose(grad_input, grads[0])
assert torch.allclose(grad_weight, grads[1])
# assert torch.allclose(grad_bias, grads[2])
