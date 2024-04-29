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

# TODO: After upgrading the software stack, test varlen flash attention op again.
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

    ouput_forward_ref, grads_ref = call_module(
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
    ouput_forward_ext, grads_ext = call_module(
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
    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-5)


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

    ouput_forward_ref, grads_ref = call_module(
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
    ouput_forward_ext, grads_ext = call_module(
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
    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-5)


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

    ouput_forward_ref, grads_ref = call_module(
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
    ouput_forward_ext, grads_ext = call_module(
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
    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-5)


def test_cross_attention_varlen_q_kv_mha():
    total_seqlen, num_heads, headdim = [16384, 6, 64]

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

    cu_seqlens_gpu = torch.tensor([  0,   186,   382,  1259,  1464,  2547,  2705,  3495,  3854,  4696,
         4762,  4885,  5118,  5355,  5503,  5760,  6168,  6353,  8272,  8461,
         9273,  9531,  9763,  9871, 10234, 10370, 10574, 10712, 11022, 11236,
        11599, 11837, 12179, 12320, 12560, 12731, 13038, 13180, 13477, 14025,
        14742, 14872, 15131, 15773, 15967, 16110, 16384], dtype=torch.int64, device='cuda')
    cu_seqlens_k_gpu = torch.tensor([  0,   186,   382,  1259,  1464,  2547,  2705,  3495,  3854,  4696,
         4762,  4885,  5118,  5355,  5503,  5760,  6168,  6353,  8272,  8461,
         9273,  9531,  9763,  9871, 10234, 10370, 10574, 10712, 11022, 11236,
        11599, 11837, 12179, 12320, 12560, 12731, 13038, 13180, 13477, 14025,
        14742, 14872, 15131, 15773, 15967, 16110, 16384], dtype=torch.int64, device='cuda')
    max_seqlen = 1919

    ouput_forward_ref, grads_ref = call_module(
        CrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )
    ouput_forward_ext, grads_ext = call_module(
        FlashCrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-5)


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

    ouput_forward_ref, grads_ref = call_module(
        CrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )
    ouput_forward_ext, grads_ext = call_module(
        FlashCrossAttention().cuda(),
        q_gpu,
        kv_gpu,
        True,
        cu_seqlens_gpu,
        max_seqlen,
        cu_seqlens_k_gpu,
        max_seqlen,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-5)
