# Copyright (c) 2024, DeepLink.

import torch

import pdb
# pdb.set_trace()

from tests.core import copy_to_cpu, allclose, call_module, call_func

from deeplink_ext.internevo_ops.flash_attention import (
    FlashSelfAttention,
    FlashCrossAttention,
)
from deeplink_ext.internevo_ops.flash_attention_fallback import (
    SelfAttention,
    CrossAttention,
)


def test_self_attention():
    batch, seqlen, nheads, headdim = [8, 32, 16, 64]

    q_gpu = torch.rand(
        [batch, seqlen, nheads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    k_gpu = torch.rand(
        [batch, seqlen, nheads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    v_gpu = torch.rand(
        [batch, seqlen, nheads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )

    q_cpu, k_cpu, v_cpu = copy_to_cpu([q_gpu, k_gpu, v_gpu])
    ouput_forward_cpu, grads_cpu = call_module(
        SelfAttention(), None, q_cpu, k_cpu, v_cpu, None
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashSelfAttention().cuda(), None, q_gpu, k_gpu, v_gpu, None
    )
    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)


def test_cross_attention():
    batch, seqlen, nheads, headdim = [8, 32, 16, 64]

    q_gpu = torch.rand(
        [batch, seqlen, nheads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    kv_gpu = torch.rand(
        [batch, seqlen, 2, nheads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )

    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])
    ouput_forward_cpu, grads_cpu = call_module(CrossAttention(), q_cpu, kv_cpu)
    ouput_forward_gpu, grads_gpu = call_module(
        FlashCrossAttention().cuda(), q_gpu, kv_gpu
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)
