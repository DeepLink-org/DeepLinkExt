# Copyright (c) 2024, DeepLink.

import torch
from tests.core import copy_to_cpu, allclose, call_module

from deeplink_ext.internevo_ops.flash_attention import (
    FlashSelfAttention,
    FlashCrossAttention,
)
from deeplink_ext.internevo_ops.flash_attention_fallback import (
    SelfAttention,
    CrossAttention,
)


def test_self_attention_varlen_qkv_mha():
    total_seqlen, num_heads, headdim = [256, 32, 64]

    qkv_gpu = torch.randn(
        [total_seqlen, 3, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    cu_seqlens_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    max_seqlen = 128

    qkv_cpu = copy_to_cpu(
        [
            qkv_gpu,
        ]
    )
    cu_seqlens_cpu = cu_seqlens_gpu.cpu()

    ouput_forward_cpu, grads_cpu = call_module(
        SelfAttention().cuda(),
        qkv_gpu,
        None,
        None,
        None,
        None,
        True,
        cu_seqlens_gpu,
        max_seqlen,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashSelfAttention().cuda(),
        qkv_gpu,
        None,
        None,
        None,
        None,
        True,
        cu_seqlens_gpu,
        max_seqlen,
    )
    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-5)


def test_self_attention_varlen_q_k_v_gqa():
    total_seqlen, num_q_heads, headdim = [256, 32, 64]
    num_kv_heads = 8

    q_gpu = torch.randn(
        [total_seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    k_gpu = torch.randn(
        [total_seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    v_gpu = torch.randn(
        [total_seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )

    cu_seqlens_q_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    max_seqlen = 128

    q_cpu, k_cpu, v_cpu = copy_to_cpu([q_gpu, k_gpu, v_gpu])
    cu_seqlens_q_cpu = cu_seqlens_q_gpu.cpu()
    cu_seqlens_k_cpu = cu_seqlens_k_gpu.cpu()

    ouput_forward_cpu, grads_cpu = call_module(
        SelfAttention().cuda(),
        None,
        q_gpu,
        k_gpu,
        v_gpu,
        None,
        True,
        None,
        None,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen,
        max_seqlen,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashSelfAttention().cuda(),
        None,
        q_gpu,
        k_gpu,
        v_gpu,
        None,
        True,
        None,
        None,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen,
        max_seqlen,
    )
    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-5)


def test_self_attention_varlen_q_kv_gqa():
    total_seqlen, num_q_heads, headdim = [256, 32, 64]
    num_kv_heads = 8

    q_gpu = torch.randn(
        [total_seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    kv_gpu = torch.randn(
        [total_seqlen, 2, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )

    cu_seqlens_q_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    max_seqlen = 128

    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])
    cu_seqlens_q_cpu = cu_seqlens_q_gpu.cpu()
    cu_seqlens_k_cpu = cu_seqlens_k_gpu.cpu()

    ouput_forward_cpu, grads_cpu = call_module(
        SelfAttention().cuda(),
        None,
        q_gpu,
        None,
        None,
        kv_gpu,
        True,
        None,
        None,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen,
        max_seqlen,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashSelfAttention().cuda(),
        None,
        q_gpu,
        None,
        None,
        kv_gpu,
        True,
        None,
        None,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen,
        max_seqlen,
    )
    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-5)


def test_cross_attention_varlen_q_kv_mha():
    total_seqlen, num_heads, headdim = [256, 32, 64]

    q_gpu = torch.randn(
        [total_seqlen, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    kv_gpu = torch.randn(
        [total_seqlen, 2, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )

    cu_seqlens_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    max_seqlen = 128

    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])
    cu_seqlens_cpu = cu_seqlens_gpu.cpu()
    cu_seqlens_k_cpu = cu_seqlens_k_gpu.cpu()

    ouput_forward_cpu, grads_cpu = call_module(
        CrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashCrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-5)


def test_cross_attention_varlen_q_kv_gqa():
    total_seqlen, num_q_heads, headdim = [256, 32, 64]
    num_kv_heads = 8

    q_gpu = torch.randn(
        [total_seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    kv_gpu = torch.randn(
        [total_seqlen, 2, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )

    cu_seqlens_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int64, device="cuda"
    )
    max_seqlen = 128

    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])
    cu_seqlens_cpu = cu_seqlens_gpu.cpu()
    cu_seqlens_k_cpu = cu_seqlens_k_gpu.cpu()

    ouput_forward_cpu, grads_cpu = call_module(
        CrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )
    ouput_forward_gpu, grads_gpu = call_module(
        FlashCrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-5)
