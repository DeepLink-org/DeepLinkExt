from DipuExt_poc.ext_apply.internlm.RMSNorm import InternLMRMSNorm, DeeplinkRMSNorm, DeeplinkRMSNorm_WithNormalizedShape
import torch
from torch import nn
import torch_dipu
import numpy as np


def test_forward_backward(Basenet, Testnet, rtol=1e-5, atol=1e-5):

    input = torch.randn(5, 5, requires_grad=True).cuda()
    input_dipu = input.clone()
    hidden_size = 5

    intern = Basenet(hidden_size).cuda()
    deep = Testnet(hidden_size).cuda()
    y_intern = intern(input)
    y_dipu = deep(input_dipu)

    y_label = torch.ones_like(y_intern)

    print("Are the prediction identical?:", np.allclose(y_intern.detach().cpu().numpy(), y_dipu.detach().cpu().numpy(), rtol, atol, True))

    loss_fn = torch.nn.MSELoss()

    loss = loss_fn(y_label, y_intern)
    input.retain_grad()
    loss.backward()
    # print("\nGradient for 'input':")
    input_grad = input.grad
    # print(input_grad)

    loss2 = loss_fn(y_label, y_dipu)
    input_dipu.retain_grad()
    loss2.backward()
    # print("\nGradient for 'input_dipu':")
    input_dipu_grad = input_dipu.grad
    # print(input_dipu_grad)

    # 对比两者是否一致
    print("Are the gradients identical?:", np.allclose(input_grad.detach().cpu().numpy(), input_dipu_grad.detach().cpu().numpy(), rtol, atol, True))


print("\nTest case: normalized_shape == None:")
test_forward_backward(InternLMRMSNorm, DeeplinkRMSNorm)
print("\nTest case: normalized_shape == weight.size():")
test_forward_backward(InternLMRMSNorm, DeeplinkRMSNorm_WithNormalizedShape)
