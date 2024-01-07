import torch
import torch_dipu
from einops import rearrange
import dipu_ext.ext_
from DeepLinkExt.ext_apply.internlm.ext_apply_rotary import (
    TorchApplyRotaryEmbQKV_,
    DeepLinkApplyRotaryEmbQKV_,
    TorchApplyRotaryEmb,
    DeepLinkApplyRotaryEmb,
)

def RotaryEmbTest(func_name):

    if func_name == "RotaryEmbQKV":
        torch_apply = TorchApplyRotaryEmbQKV_.apply
        dipu_apply = DeepLinkApplyRotaryEmbQKV_.apply
        input = torch.randn(1, 125, 3, 16, 32, dtype=torch.float16, requires_grad=True).cuda()
    elif func_name == "RotaryEmb":
        torch_apply = TorchApplyRotaryEmb.apply
        dipu_apply = DeepLinkApplyRotaryEmb.apply
        input = torch.randn(1, 125, 16, 32, dtype=torch.float16, requires_grad=True).cuda()
    else:
         print(f"{func_name} is not supported.")
         return False

    loss_fn = torch.nn.MSELoss()
    cos = torch.randn(257, 16, dtype=torch.float16).cuda()
    sin = torch.randn(257, 16, dtype=torch.float16).cuda()
    input1 = input.detach().clone()
    input1.requires_grad = True
    cos1 = cos.clone()
    sin1 = sin.clone()
    cos_k = None
    sin_k = None
    interleaved = False

    # 调用前向传播
    if func_name == "RotaryEmbQKV":
        res1 = torch_apply(input, cos, sin, cos_k, sin_k, interleaved)
        res2 = dipu_apply(input1, cos1, sin1, cos_k, sin_k, interleaved)
    elif func_name == "RotaryEmb":
        res1 = torch_apply(input, cos, sin, interleaved)
        res2 = dipu_apply(input1, cos1, sin1, interleaved)
    else:
         print(f"{func_name} is not supported.")
         return False


    # 验证前向传播结果
    forward_correct = torch.allclose(res1, res2)

    # 计算第一个损失
    c = torch.ones_like(res1)
    loss1 = loss_fn(res1, c)  # 将输出的元素求和，得到标量
    input.retain_grad()
    loss1.backward()

    # 计算第二个损失
    c2 = torch.ones_like(res1)
    loss2 = loss_fn(res2, c2)  # 将输出的元素求和，得到标量
    input1.retain_grad()
    loss2.backward()

    # 验证第一个反向传播梯度
    grad1 = input.grad
    grad2 = input1.grad
    backward_correct = torch.allclose(grad1, grad2)
    # 判断前向和反向传播结果是否都正确
    if forward_correct and backward_correct:
        print(f"{func_name} both forward and backward pass tests passed.")
        return True
    else:
        print(
            f"{func_name} tests failed: Forward pass:",
            forward_correct,
            "Backward pass:",
            backward_correct,
        )
        return False

assert RotaryEmbTest("RotaryEmbQKV")
assert RotaryEmbTest("RotaryEmb")
