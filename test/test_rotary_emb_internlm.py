import torch
import torch_dipu
from einops import rearrange
import dipu_ext.ext_
from DipuExt_poc.ext_apply.internlm.ext_apply_rotary import TorchApplyRotaryEmbQKV_, DeeplLinkApplyRotaryEmbQKV_

torch_apply = TorchApplyRotaryEmbQKV_.apply
dipu_apply = DeeplLinkApplyRotaryEmbQKV_.apply
loss_fn = torch.nn.MSELoss()

# 创建输入数据
qkv = torch.randn(1, 125, 3, 16, 32, dtype=torch.float16, requires_grad=True).cuda()
cos = torch.randn(257, 16, dtype=torch.float16).cuda()
sin = torch.randn(257, 16, dtype=torch.float16).cuda()
qkv1 = qkv.detach().clone()
qkv1.requires_grad=True
cos1 = cos.clone()
sin1 = sin.clone()
cos_k = None
sin_k = None
interleaved = False

# 调用前向传播
res1 = torch_apply(qkv, cos, sin, cos_k, sin_k, interleaved)
res2 = dipu_apply(qkv1, cos1, sin1, cos_k, sin_k, interleaved)

# 验证前向传播结果
forward_correct = torch.allclose(res1, res2)

# 计算第一个损失
c = torch.ones_like(res1)
loss1 = loss_fn(res1, c)  # 将输出的元素求和，得到标量
qkv.retain_grad()
loss1.backward()

# 计算第二个损失
c2 = torch.ones_like(res1)
loss2 = loss_fn(res2, c2)  # 将输出的元素求和，得到标量
qkv1.retain_grad()
loss2.backward()

# 验证第一个反向传播梯度
grad1 = qkv.grad
grad2 = qkv1.grad
backward_correct = torch.allclose(grad1, grad2)
# 判断前向和反向传播结果是否都正确
if forward_correct and backward_correct:
    print("Both forward and backward pass tests passed.")
else:
    print("Tests failed: Forward pass:", forward_correct, "Backward pass:", backward_correct)
