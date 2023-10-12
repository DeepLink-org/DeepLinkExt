from ext_apply.internlm.RMSNorm import InternLMRMSNorm, DeeplinkRMSNorm
import torch
from torch import nn
import torch_dipu
import numpy as np



input = torch.randn(5, 5, requires_grad=True).cuda()
input_dipu = input.clone()
hidden_size = 5


intern = InternLMRMSNorm(hidden_size).cuda()
deep = DeeplinkRMSNorm(hidden_size).cuda()
a = intern(input)
b = deep(input_dipu)
print(a)
print(b)

c = torch.ones_like(a)


# 1. 定义一个简单的损失函数
loss_fn = torch.nn.MSELoss()

loss = loss_fn(a, c)
input.retain_grad()
loss.backward()
print("\nGradient for 'input':")
input_grad = input.grad
print(input_grad)

loss2 = loss_fn(b, c)
input_dipu.retain_grad()
loss2.backward()
print("\nGradient for 'input_dipu':")
input_dipu_grad = input_dipu.grad
print(input_dipu_grad)

# 对比两者是否一致
rtol = 1e-5
atol = 1e-5
print("\nAre the gradients identical?:", np.allclose(input_grad.detach().cpu().numpy(), input_dipu_grad.detach().cpu().numpy(), rtol, atol, True))