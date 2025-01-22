# Copyright (c) 2024, DeepLink.

import torch
import torch_npu

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
]


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    head_num = q.shape[-2]

    assert seqlen_q == seqlen_k
    sparse_mode = 2 if seqlen_q > 2048 else 0
    seqlen = min(seqlen_q, 2048)

    attention_mask = (
        torch.triu(
            torch.ones([seqlen, seqlen], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=seqlen_q,
        next_tockens=0,
        sparse_mode=sparse_mode,
    )[0]

    return out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_num = q.shape[-2]

    cu_seqlens_q = cu_seqlens_q[1:].tolist()
    cu_seqlens_k = cu_seqlens_k[1:].tolist()

    assert max_seqlen_q == max_seqlen_k
    sparse_mode = 2 if max_seqlen_q > 2048 else 0
    max_seqlen = min(max_seqlen_q, 2048)

    attention_mask = (
        torch.triu(
            torch.ones([max_seqlen, max_seqlen], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=q.shape[0],
        next_tockens=0,
        keep_prob=1 - dropout_p,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,
    )[0]
    return out


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** (-0.5)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]

    seqlen_qkv = qkv.shape[1]
    head_num = q.shape[-2]

    sparse_mode = 2 if seqlen_qkv > 2048 else 0
    seqlen = min(seqlen_qkv, 2048)

    attention_mask = (
        torch.triu(
            torch.ones([seqlen, seqlen], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=seqlen_qkv,
        next_tockens=0,
        sparse_mode=sparse_mode,
    )[0]

    return out


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    k = kv[:, :, 0]
    v = kv[:, :, 1]

    seqlen_q = q.shape[1]
    seqlen_kv = kv.shape[1]
    head_num = q.shape[-2]

    assert seqlen_q == seqlen_kv
    sparse_mode = 2 if seqlen_q > 2048 else 0
    seqlen = min(seqlen_q, 2048)

    attention_mask = (
        torch.triu(
            torch.ones([seqlen, seqlen], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=seqlen_q,
        next_tockens=0,
        sparse_mode=sparse_mode,
    )[0]

    return out


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** (-0.5)
    q = qkv[:, 0]
    k = qkv[:, 1]
    v = qkv[:, 2]
    head_num = q.shape[1]

    cu_seqlens_q = cu_seqlens[1:].tolist()
    cu_seqlens_k = cu_seqlens[1:].tolist()

    sparse_mode = 2 if max_seqlen > 2048 else 0
    max_seqlen = min(max_seqlen, 2048)
    attention_mask = (
        torch.triu(
            torch.ones([max_seqlen, max_seqlen], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=q.shape[0],
        next_tockens=0,
        keep_prob=1 - dropout_p,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,
    )[0]
    return out


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    k = kv[:, 0]
    v = kv[:, 1]
    head_num = q.shape[1]
    cu_seqlens_q = cu_seqlens_q[1:].tolist()
    cu_seqlens_k = cu_seqlens_k[1:].tolist()

    assert max_seqlen_q == max_seqlen_k
    sparse_mode = 2 if max_seqlen_q > 2048 else 0
    max_seqlen = min(max_seqlen_q, 2048)

    attention_mask = (
        torch.triu(
            torch.ones([max_seqlen, max_seqlen], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        if causal
        else None
    )

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=q.shape[0],
        next_tockens=0,
        keep_prob=1 - dropout_p,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,
    )[0]
    return out
