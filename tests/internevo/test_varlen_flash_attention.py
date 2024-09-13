# Copyright (c) 2024, DeepLink.

import torch
from tests.core import allclose, calculate_fwd_and_bwd, copy_to_cpu

from deeplink_ext.internevo_ops.flash_attention_fallback import (
    flash_attn_varlen_qkvpacked_func_torch,
    flash_attn_varlen_kvpacked_func_torch,
    flash_attn_varlen_func_torch,
)
from deeplink_ext.internevo_ops.flash_attention import (
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_func,
)


# fmt: off
# latest sequence length is 20206-16110=4096
cu_seqlens_max_length_4096 = [
    0, 186, 382, 1259, 1464, 2547, 2705, 3495, 3854, 4696, 4762, 4885, 5118, 5355, 5503, 5760, 6168, 6353,
    8272, 8461, 9273, 9531, 9763, 9871, 10234, 10370, 10574, 10712, 11022, 11236, 11599, 11837, 12179, 12320,
    12560, 12731, 13038, 13180, 13477, 14025, 14742, 14872, 15131, 15773, 15967, 16110, 20206,
]
# fmt: on


def test_flash_attn_varlen_qkvpacked_func_mha():
    total_seqlen, num_heads, headdim = [256, 32, 64]

    qkv_gpu = torch.randn(
        [total_seqlen, 3, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    qkv_cpu = copy_to_cpu(
        [
            qkv_gpu,
        ]
    )

    cu_seqlens_cpu = torch.tensor([0, 32, 64, 128, 256], dtype=torch.int32)
    cu_seqlens_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    max_seqlen = 128

    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_varlen_qkvpacked_func_torch,
        qkv_cpu[0],
        cu_seqlens_cpu,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_varlen_qkvpacked_func,
        qkv_gpu,
        cu_seqlens_gpu,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-2)


def test_flash_attn_varlen_qkvpacked_func_mha_long_max_seqlen():
    # Test function to verify if the module behaves correctly when the maximum sequence length exceeds 2048.
    total_seqlen, num_heads, headdim = [20206, 2, 64]

    qkv_gpu = torch.randn(
        [total_seqlen, 3, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    qkv_cpu = copy_to_cpu(
        [
            qkv_gpu,
        ]
    )

    cu_seqlens_cpu = torch.tensor(cu_seqlens_max_length_4096, dtype=torch.int32)
    cu_seqlens_gpu = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    # the maximum sequence length is 4096
    max_seqlen = 4096

    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_varlen_qkvpacked_func_torch,
        qkv_cpu[0],
        cu_seqlens_cpu,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_varlen_qkvpacked_func,
        qkv_gpu,
        cu_seqlens_gpu,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-2)


def test_flash_attn_varlen_kvpacked_func_gqa():
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
    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])

    cu_seqlens_q_cpu = torch.tensor([0, 32, 64, 128, 256], dtype=torch.int32)
    cu_seqlens_k_cpu = torch.tensor([0, 32, 64, 128, 256], dtype=torch.int32)
    cu_seqlens_q_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    max_seqlen_q = 128
    max_seqlen_k = 128

    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_varlen_kvpacked_func_torch,
        q_cpu,
        kv_cpu,
        cu_seqlens_q_cpu,
        cu_seqlens_k_cpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_varlen_kvpacked_func,
        q_gpu,
        kv_gpu,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-2)


def test_flash_attn_varlen_kvpacked_func_gqa_long_max_seqlen():
    # Test function to verify if the module behaves correctly when the maximum sequence length exceeds 2048.
    total_seqlen, num_q_heads, headdim = [20206, 6, 64]
    num_kv_heads = 2

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
    q_cpu, kv_cpu = copy_to_cpu([q_gpu, kv_gpu])

    cu_seqlens_q_cpu = torch.tensor(cu_seqlens_max_length_4096, dtype=torch.int32)
    cu_seqlens_k_cpu = torch.tensor(cu_seqlens_max_length_4096, dtype=torch.int32)
    cu_seqlens_q_gpu = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    # the maximum sequence length is 4096
    max_seqlen_q = 4096
    max_seqlen_k = 4096

    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_varlen_kvpacked_func_torch,
        q_cpu,
        kv_cpu,
        cu_seqlens_q_cpu,
        cu_seqlens_k_cpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_varlen_kvpacked_func,
        q_gpu,
        kv_gpu,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-2)


def test_flash_attn_varlen_func_gqa():
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
    q_cpu, k_cpu, v_cpu = copy_to_cpu([q_gpu, k_gpu, v_gpu])

    cu_seqlens_q_cpu = torch.tensor([0, 32, 64, 128, 256], dtype=torch.int32)
    cu_seqlens_k_cpu = torch.tensor([0, 32, 64, 128, 256], dtype=torch.int32)
    cu_seqlens_q_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    max_seqlen_q = 128
    max_seqlen_k = 128

    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_varlen_func_torch,
        q_cpu,
        k_cpu,
        v_cpu,
        cu_seqlens_q_cpu,
        cu_seqlens_k_cpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_varlen_func,
        q_gpu,
        k_gpu,
        v_gpu,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-2)


def test_flash_attn_varlen_func_gqa_long_max_seqlen():
    # Test function to verify if the module behaves correctly when the maximum sequence length exceeds 2048.
    total_seqlen, num_q_heads, headdim = [20206, 6, 64]
    num_kv_heads = 2

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
    q_cpu, k_cpu, v_cpu = copy_to_cpu([q_gpu, k_gpu, v_gpu])

    cu_seqlens_q_cpu = torch.tensor(cu_seqlens_max_length_4096, dtype=torch.int32)
    cu_seqlens_k_cpu = torch.tensor(cu_seqlens_max_length_4096, dtype=torch.int32)
    cu_seqlens_q_gpu = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_gpu = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    # the maximum sequence length is 4096
    max_seqlen_q = 4096
    max_seqlen_k = 4096

    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_varlen_func_torch,
        q_cpu,
        k_cpu,
        v_cpu,
        cu_seqlens_q_cpu,
        cu_seqlens_k_cpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_varlen_func,
        q_gpu,
        k_gpu,
        v_gpu,
        cu_seqlens_q_gpu,
        cu_seqlens_k_gpu,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-5, atol=1e-5)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-5, atol=1e-2)
