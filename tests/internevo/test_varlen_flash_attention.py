# Copyright (c) 2024, DeepLink.

import torch
from tests.core import allclose, call_normal_func

from deeplink_ext.internevo_ops.flash_attention_fallback import (
    torch_attn_varlen_qkvpacked_func,
    torch_attn_varlen_kvpacked_func,
    torch_attn_varlen_func,
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

    qkv_ref = torch.randn(
        [total_seqlen, 3, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    qkv_ext = qkv_ref.clone().detach().requires_grad_(True)

    cu_seqlens_ref = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    max_seqlen = 128

    ouput_forward_ref, grads_ref = call_normal_func(
        torch_attn_varlen_qkvpacked_func,
        qkv_ref,
        cu_seqlens_ref,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_ext, grads_ext = call_normal_func(
        flash_attn_varlen_qkvpacked_func,
        qkv_ext,
        cu_seqlens_ref,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-2)


def test_flash_attn_varlen_qkvpacked_func_mha_long_max_seqlen():
    # Test function to verify if the module behaves correctly when the maximum sequence length exceeds 2048.
    total_seqlen, num_heads, headdim = [20206, 2, 64]

    qkv_ref = torch.randn(
        [total_seqlen, 3, num_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    qkv_ext = qkv_ref.clone().detach().requires_grad_(True)

    cu_seqlens_ref = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    # the maximum sequence length is 4096
    max_seqlen = 4096

    ouput_forward_ref, grads_ref = call_normal_func(
        torch_attn_varlen_qkvpacked_func,
        qkv_ref,
        cu_seqlens_ref,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_ext, grads_ext = call_normal_func(
        flash_attn_varlen_qkvpacked_func,
        qkv_ext,
        cu_seqlens_ref,
        max_seqlen,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-2)


def test_flash_attn_varlen_kvpacked_func_gqa():
    total_seqlen, num_q_heads, headdim = [256, 32, 64]
    num_kv_heads = 8

    q_ref = torch.randn(
        [total_seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    kv_ref = torch.randn(
        [total_seqlen, 2, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    q_ext = q_ref.clone().detach().requires_grad_(True)
    kv_ext = kv_ref.clone().detach().requires_grad_(True)

    cu_seqlens_q_ref = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_ref = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    max_seqlen_q = 128
    max_seqlen_k = 128

    ouput_forward_ref, grads_ref = call_normal_func(
        torch_attn_varlen_kvpacked_func,
        q_ref,
        kv_ref,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_ext, grads_ext = call_normal_func(
        flash_attn_varlen_kvpacked_func,
        q_ext,
        kv_ext,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-3, atol=1e-2)


def test_flash_attn_varlen_kvpacked_func_gqa_long_max_seqlen():
    # Test function to verify if the module behaves correctly when the maximum sequence length exceeds 2048.
    total_seqlen, num_q_heads, headdim = [20206, 6, 64]
    num_kv_heads = 2

    q_ref = torch.randn(
        [total_seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    kv_ref = torch.randn(
        [total_seqlen, 2, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    q_ext = q_ref.clone().detach().requires_grad_(True)
    kv_ext = kv_ref.clone().detach().requires_grad_(True)

    cu_seqlens_q_ref = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_ref = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    # the maximum sequence length is 4096
    max_seqlen_q = 4096
    max_seqlen_k = 4096

    ouput_forward_ref, grads_ref = call_normal_func(
        torch_attn_varlen_kvpacked_func,
        q_ref,
        kv_ref,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_ext, grads_ext = call_normal_func(
        flash_attn_varlen_kvpacked_func,
        q_ext,
        kv_ext,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-3, atol=1e-2)


def test_flash_attn_varlen_func_gqa():
    total_seqlen, num_q_heads, headdim = [256, 32, 64]
    num_kv_heads = 8

    q_ref = torch.randn(
        [total_seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    k_ref = torch.randn(
        [total_seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    v_ref = torch.randn(
        [total_seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    q_ext = q_ref.clone().detach().requires_grad_(True)
    k_ext = k_ref.clone().detach().requires_grad_(True)
    v_ext = v_ref.clone().detach().requires_grad_(True)

    cu_seqlens_q_ref = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_ref = torch.tensor(
        [0, 32, 64, 128, 256], dtype=torch.int32, device="cuda"
    )
    max_seqlen_q = 128
    max_seqlen_k = 128

    ouput_forward_ref, grads_ref = call_normal_func(
        torch_attn_varlen_func,
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_ext, grads_ext = call_normal_func(
        flash_attn_varlen_func,
        q_ext,
        k_ext,
        v_ext,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-2)


def test_flash_attn_varlen_func_gqa_long_max_seqlen():
    # Test function to verify if the module behaves correctly when the maximum sequence length exceeds 2048.
    total_seqlen, num_q_heads, headdim = [20206, 6, 64]
    num_kv_heads = 2

    q_ref = torch.randn(
        [total_seqlen, num_q_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    k_ref = torch.randn(
        [total_seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    v_ref = torch.randn(
        [total_seqlen, num_kv_heads, headdim],
        dtype=torch.float16,
        requires_grad=True,
        device="cuda",
    )
    q_ext = q_ref.clone().detach().requires_grad_(True)
    k_ext = k_ref.clone().detach().requires_grad_(True)
    v_ext = v_ref.clone().detach().requires_grad_(True)

    cu_seqlens_q_ref = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k_ref = torch.tensor(
        cu_seqlens_max_length_4096, dtype=torch.int32, device="cuda"
    )
    # the maximum sequence length is 4096
    max_seqlen_q = 4096
    max_seqlen_k = 4096

    ouput_forward_ref, grads_ref = call_normal_func(
        torch_attn_varlen_func,
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    ouput_forward_ext, grads_ext = call_normal_func(
        flash_attn_varlen_func,
        q_ext,
        k_ext,
        v_ext,
        cu_seqlens_q_ref,
        cu_seqlens_k_ref,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )

    assert allclose(ouput_forward_ref, ouput_forward_ext, rtol=1e-5, atol=1e-5)
    assert allclose(grads_ref, grads_ext, rtol=1e-5, atol=1e-2)
