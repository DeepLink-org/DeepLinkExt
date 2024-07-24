# Copyright (c) 2024, DeepLink.
from einops import rearrange, repeat
import torch
from torch.nn.utils.rnn import pad_sequence
import math


__all__ = [
    "torch_attn_qkvpacked_func",
    "torch_attn_kvpacked_func",
    "torch_attn_func",
    "torch_attn_varlen_qkvpacked_func",
    "torch_attn_varlen_kvpacked_func",
    "torch_attn_varlen_func",
]


def _unpack_qkv_before_attn(
    cur_input: torch.Tensor, cu_seqlens: torch.Tensor, padding_v: int = 0
):
    """
    qkv: the shape is (packed_length, three, head_num, head_dim)
    kv: the shape is (packed_length, two, head_num, head_dim)
    q/k/v: the shape is (packed_length, head_num, head_dim)

    Return:
    output: the shape is (micro_bsz, seq_len, three, head_num, head_dim) for qkv
                        (micro_bsz, seq_len, two, head_num, head_dim) for kv
                        (micro_bsz, seq_len, head_num, head_dim) for q/k/v
    """
    sequences = []
    for i in range(len(cu_seqlens) - 1):
        sequences.append(cur_input[cu_seqlens[i] : cu_seqlens[i + 1]])

    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=padding_v
    )

    return padded_sequences


def _pack_output_after_attn(
    cur_input: torch.Tensor,
    cu_seqlens: torch.Tensor,
    packed_length: int,
    padding_v: int = 0,
):
    """
    cur_input: the shape is (micro_bsz, max_seq_len, head_num, head_dim)

    Return:
    output: the shape is (packed_length, head_num, head_dim)
    """
    output_shape = [packed_length, *cur_input.shape[-2:]]

    output = torch.full(
        output_shape,
        fill_value=padding_v,
        device=cur_input.device,
        dtype=cur_input.dtype,
    )
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        output[cu_seqlens[i] : cu_seqlens[i + 1]] = cur_input[i, 0:length]

    return output


def torch_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    seqlen = qkv.shape[1]
    q, k, v = qkv.unbind(dim=2)

    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

    if causal:
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)

    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    dropout = torch.nn.Dropout(dropout_p)
    attention_drop = dropout(attention)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    return output


def torch_attn_kvpacked_func(
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
    batch_size, seqlen_q = q.shape[0], q.shape[1]
    seqlen_k = kv.shape[1]

    assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
    if kv.shape[3] != q.shape[2]:  # MQA/GQA
        kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
    k, v = kv.unbind(dim=2)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

    if causal:
        row_idx = rearrange(
            torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1"
        )
        col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
        causal_mask = col_idx > row_idx + seqlen_k - seqlen_q
        scores = scores.masked_fill(causal_mask, -10000.0)

    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    dropout = torch.nn.Dropout(dropout_p)
    attention_drop = dropout(attention)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    return output


def torch_attn_func(
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
    kv = torch.stack([k, v], dim=2)
    return torch_attn_kvpacked_func(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def torch_attn_varlen_qkvpacked_func(
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
    packed_length = qkv.size(dim=0)
    qkv = _unpack_qkv_before_attn(qkv, cu_seqlens=cu_seqlens)
    output = torch_attn_qkvpacked_func(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )
    return _pack_output_after_attn(output, cu_seqlens, packed_length)


def torch_attn_varlen_kvpacked_func(
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
    packed_length = q.size(dim=0)
    q = _unpack_qkv_before_attn(q, cu_seqlens=cu_seqlens_q)
    kv = _unpack_qkv_before_attn(kv, cu_seqlens=cu_seqlens_k)
    output = torch_attn_kvpacked_func(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )
    return _pack_output_after_attn(output, cu_seqlens_q, packed_length)


def torch_attn_varlen_func(
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
    block_table=None,
):
    packed_length = q.size(dim=0)
    kv = torch.stack([k, v], dim=1)
    q = _unpack_qkv_before_attn(q, cu_seqlens=cu_seqlens_q)
    kv = _unpack_qkv_before_attn(kv, cu_seqlens=cu_seqlens_k)
    output = torch_attn_kvpacked_func(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )
    return _pack_output_after_attn(output, cu_seqlens_q, packed_length)
