# Copyright (c) 2024, DeepLink.

import torch
import torch_npu
from tests.core import copy_to_cpu, allclose, call_module

from deeplink_ext.interntrain_ops.flash_attention import (
    FlashSelfAttention,
    FlashCrossAttention,
)
from deeplink_ext.interntrain_ops.flash_attention_fallback import (
    SelfAttention,
    CrossAttention,
)


def test_self_attention_qkv_mha():
    batch, seqlen, num_heads, headdim = [8, 32, 32, 64]

    qkv_gpu = torch.rand(
        [batch, seqlen, 3, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )

    qkv_cpu = copy_to_cpu([
        qkv_gpu,
    ])
    ouput_forward_cpu, grads_cpu = call_module(
        SelfAttention(),
        qkv_cpu[0],
        None,
        None,
        None,
        None,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashSelfAttention().npu(),
        qkv_gpu,
        None,
        None,
        None,
        None,
    )
    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)


def test_self_attention_q_k_v_gqa():
    batch, seqlen, num_q_heads, headdim = [8, 32, 32, 64]
    num_kv_heads = 8

    q_gpu = torch.rand(
        [batch, seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )
    k_gpu = torch.rand(
        [batch, seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )
    v_gpu = torch.rand(
        [batch, seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )

    q_cpu, k_cpu, v_cpu = copy_to_cpu([q_gpu, k_gpu, v_gpu])
    ouput_forward_cpu, grads_cpu = call_module(
        SelfAttention(),
        None,
        q_cpu,
        k_cpu,
        v_cpu,
        None,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashSelfAttention().npu(),
        None,
        q_gpu,
        k_gpu,
        v_gpu,
        None,
    )
    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)


def test_self_attention_q_kv_gqa():
    batch, seqlen, num_q_heads, headdim = [8, 32, 32, 64]
    num_kv_heads = 8

    q_gpu = torch.rand(
        [batch, seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )
    kv_gpu = torch.rand(
        [batch, seqlen, 2, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )

    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])
    ouput_forward_cpu, grads_cpu = call_module(
        SelfAttention(),
        None,
        q_cpu,
        None,
        None,
        kv_cpu,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashSelfAttention().npu(),
        None,
        q_gpu,
        None,
        None,
        kv_gpu,
    )
    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)


def test_cross_attention_q_kv_mha():
    batch, seqlen, num_heads, headdim = [8, 32, 32, 64]

    q_gpu = torch.rand(
        [batch, seqlen, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )
    kv_gpu = torch.rand(
        [batch, seqlen, 2, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )

    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])
    ouput_forward_cpu, grads_cpu = call_module(
        CrossAttention(),
        q_cpu,
        kv_cpu,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashCrossAttention().npu(),
        q_gpu,
        kv_gpu,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)


def test_cross_attention_q_kv_gqa():
    batch, seqlen, num_q_heads, headdim = [8, 32, 32, 64]
    num_kv_heads = 8
    q_gpu = torch.rand(
        [batch, seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )
    kv_gpu = torch.rand(
        [batch, seqlen, 2, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )

    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])
    ouput_forward_cpu, grads_cpu = call_module(CrossAttention(), q_cpu, kv_cpu)
    ouput_forward_gpu, grads_gpu = call_module(
        FlashCrossAttention().npu(),
        q_gpu,
        kv_gpu,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)
