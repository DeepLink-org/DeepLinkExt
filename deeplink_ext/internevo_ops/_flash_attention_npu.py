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

_GLOBAL_ATTN_MASK = None


def set_attention_mask(attn_mask):
    global _GLOBAL_ATTN_MASK
    _GLOBAL_ATTN_MASK = attn_mask


def get_attention_mask(seqlen, causal, window_size):
    global _GLOBAL_ATTN_MASK

    if _GLOBAL_ATTN_MASK is not None:
        return _GLOBAL_ATTN_MASK

    # causal attention
    if causal:
        if seqlen > 2048:
            _GLOBAL_ATTN_MASK = torch.triu(
                torch.ones([2048, 2048], dtype=bool, device=torch.npu.current_device()),
                diagonal=1,
            )
        else:
            _GLOBAL_ATTN_MASK = torch.triu(
                torch.ones(
                    [seqlen, seqlen], dtype=bool, device=torch.npu.current_device()
                ),
                diagonal=1,
            )

    # sliding window attention
    if window_size[0] >= 0 or window_size[1] >= 0:
        _GLOBAL_ATTN_MASK = torch.tril(
            torch.ones([seqlen, seqlen], dtype=bool, device=torch.npu.current_device()),
            diagonal=-((seqlen - 1 if window_size[0] < 0 else window_size[0]) + 1),
        ) + torch.triu(
            torch.ones([seqlen, seqlen], dtype=bool, device=torch.npu.current_device()),
            diagonal=(seqlen - 1 if window_size[1] < 0 else window_size[1]) + 1,
        )

    return _GLOBAL_ATTN_MASK


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
    set_attention_mask(None)
    attention_mask = get_attention_mask(seqlen_q, causal, window_size)
    sparse_mode = 0 if attention_mask is None or seqlen_q <= 2048 else 4

    pre_tokens = seqlen_q - 1
    next_tokens = 0
    if window_size[0] >= 0 or window_size[1] >= 0:
        pre_tokens = seqlen_q - 1 if window_size[0] < 0 else window_size[0]
        next_tokens = seqlen_q - 1 if window_size[1] < 0 else window_size[1]

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
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
    block_table=None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_num = q.shape[-2]

    cu_seqlens_q = cu_seqlens_q[1:].tolist()
    cu_seqlens_k = cu_seqlens_k[1:].tolist()

    assert max_seqlen_q == max_seqlen_k
    set_attention_mask(None)
    attention_mask = get_attention_mask(max_seqlen_q, causal, window_size)
    sparse_mode = 0 if attention_mask is None or max_seqlen_q <= 2048 else 4

    pre_tokens = max_seqlen_q - 1
    next_tokens = 0
    if window_size[0] >= 0 or window_size[1] >= 0:
        pre_tokens = max_seqlen_q - 1 if window_size[0] < 0 else window_size[0]
        next_tokens = max_seqlen_q - 1 if window_size[1] < 0 else window_size[1]

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
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

    set_attention_mask(None)
    attention_mask = get_attention_mask(seqlen_qkv, causal, window_size)
    sparse_mode = 0 if attention_mask is None or seqlen_qkv <= 2048 else 4

    pre_tokens = seqlen_qkv - 1
    next_tokens = 0
    if window_size[0] >= 0 or window_size[1] >= 0:
        pre_tokens = seqlen_qkv - 1 if window_size[0] < 0 else window_size[0]
        next_tokens = seqlen_qkv - 1 if window_size[1] < 0 else window_size[1]

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
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
    set_attention_mask(None)
    attention_mask = get_attention_mask(seqlen_q, causal, window_size)
    sparse_mode = 0 if attention_mask is None or seqlen_q <= 2048 else 4

    pre_tokens = seqlen_q - 1
    next_tokens = 0
    if window_size[0] >= 0 or window_size[1] >= 0:
        pre_tokens = seqlen_q - 1 if window_size[0] < 0 else window_size[0]
        next_tokens = seqlen_q - 1 if window_size[1] < 0 else window_size[1]

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        keep_prob=1 - dropout_p,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
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

    set_attention_mask(None)
    attention_mask = get_attention_mask(max_seqlen, causal, window_size)
    sparse_mode = 0 if attention_mask is None or max_seqlen <= 2048 else 4

    pre_tokens = max_seqlen - 1
    next_tokens = 0
    if window_size[0] >= 0 or window_size[1] >= 0:
        pre_tokens = max_seqlen - 1 if window_size[0] < 0 else window_size[0]
        next_tokens = max_seqlen - 1 if window_size[1] < 0 else window_size[1]

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
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
    set_attention_mask(None)
    attention_mask = get_attention_mask(max_seqlen_q, causal, window_size)
    sparse_mode = 0 if attention_mask is None or max_seqlen_q <= 2048 else 4

    pre_tokens = max_seqlen_q - 1
    next_tokens = 0
    if window_size[0] >= 0 or window_size[1] >= 0:
        pre_tokens = max_seqlen_q - 1 if window_size[0] < 0 else window_size[0]
        next_tokens = max_seqlen_k - 1 if window_size[1] < 0 else window_size[1]

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "TND",
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        keep_prob=1 - dropout_p,
        sparse_mode=sparse_mode,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_k,
    )[0]
    return out
