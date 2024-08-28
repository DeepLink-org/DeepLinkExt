# Copyright (c) 2024, DeepLink.

import torch
from tests.core import copy_to_cpu, allclose, calculate_fwd_and_bwd

from deeplink_ext.internevo_ops.flash_attention_fallback import (
    flash_attn_qkvpacked_func_torch,
    flash_attn_kvpacked_func_torch,
    flash_attn_func_torch,
)
from deeplink_ext.internevo_ops.flash_attention import (
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_func,
)


def test_flash_attn_qkvpacked_func_mha():
    batch, seqlen, num_heads, headdim = [8, 32, 32, 64]

    qkv_gpu = torch.rand(
        [batch, seqlen, 3, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="npu",
    )

    qkv_cpu = copy_to_cpu(
        [
            qkv_gpu,
        ]
    )

    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_qkvpacked_func_torch,
        qkv_cpu[0],
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_qkvpacked_func,
        qkv_gpu,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)


def test_flash_attn_kvpacked_func_gqa():
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
    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_kvpacked_func_torch,
        q_cpu,
        kv_cpu,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_kvpacked_func,
        q_gpu,
        kv_gpu,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)


def test_flash_attn_func_gqa():
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
    ouput_forward_cpu, grads_cpu = calculate_fwd_and_bwd(
        flash_attn_func_torch,
        q_cpu,
        k_cpu,
        v_cpu,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_gpu, grads_gpu = calculate_fwd_and_bwd(
        flash_attn_func,
        q_gpu,
        k_gpu,
        v_gpu,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_cpu, ouput_forward_gpu, rtol=1e-3, atol=1e-3)
    assert allclose(grads_cpu, grads_gpu, rtol=1e-3, atol=1e-3)
