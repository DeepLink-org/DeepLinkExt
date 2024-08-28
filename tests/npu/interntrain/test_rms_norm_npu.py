# Copyright (c) 2024, DeepLink.

import torch

from deeplink_ext.interntrain_ops.rms_norm import MixedFusedRMSNorm
from deeplink_ext.interntrain_ops.rms_norm_fallback import MixedRMSNormTorch


def test_rms_norm_npu():
    input_dtype_list = [torch.float32, torch.bfloat16, torch.float32, torch.float32]
    weight_dtype_list = [torch.float32, torch.bfloat16, torch.float16, torch.bfloat16]

    for input_dtype, weight_dtype in zip(input_dtype_list, weight_dtype_list):
        x = torch.randn(1, 64, 32, 64, dtype=input_dtype).requires_grad_()
        m_cpu = MixedRMSNormTorch([x.shape[-1]], 1e-5).to(weight_dtype)
        out = m_cpu(x)
        out.backward(torch.ones_like(out))

        y = x.detach().clone().npu().requires_grad_()
        m_npu = MixedFusedRMSNorm([y.shape[-1]], 1e-5).npu().to(weight_dtype)
        out2 = m_npu(y)

        out2.backward(torch.ones_like(out2))

        torch.allclose(out, out2.cpu(), atol=1e-4, rtol=1e-4)
        torch.allclose(x.grad, y.grad.cpu(), atol=1e-4, rtol=1e-4)
