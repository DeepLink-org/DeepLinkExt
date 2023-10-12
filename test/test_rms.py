import torch

import dipu_ext.ext_ as deeplink_ext
import pdb
# import debugat

# 假设 deeplink_ext 是一个包含上述 RMS normalization 函数的模块

# 定义输入张量
input = torch.randn(5, 5, requires_grad=True).cuda()

# 定义权重和偏差张量
weight = torch.randn(5, requires_grad=True).cuda()
bias = torch.randn(5, requires_grad=True).cuda()

# 定义输出梯度张量
grad_output = torch.randn(5, 5).cuda()

# 归一化的形状通常是输入张量的形状
normalized_shape = torch.tensor([5, 5], dtype=torch.long).cuda()

# pdb.set_trace()

# 使用 RMS normalization 前向传播
# while True:
output, inv_rms = deeplink_ext.rms_norm(
    input,
    None,
    weight,
    bias,
    1e-6
)

# 使用 RMS normalization 反向传播
grad_input, grad_weight, grad_bias = deeplink_ext.rms_norm_backward(
    input,
    grad_output,
    inv_rms,
    None,
    weight,
    bias,
    1e-6
)

print("Output:", output)
print("Grad Input:", grad_input)
print("Grad Weight:", grad_weight)
print("Grad Bias:", grad_bias)
b = input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
assert torch.allclose(output, b)
