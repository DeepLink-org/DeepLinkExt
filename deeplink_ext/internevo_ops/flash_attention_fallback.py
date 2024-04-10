# Copyright (c) 2024, DeepLink.
# Copyright (c) 2024, InternEvo.
# Copyright (c) 2023, Tri Dao.

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


__all__ = ["SelfAttention", "CrossAttention"]


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        qkv=None,
        q=None,
        k=None,
        v=None,
        kv=None,
        causal=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        softmax_scale=None,
        dropout_p=0.0
    ):
        """Only supports the padded mode"""
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        if qkv is not None:
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            device = query.device
        elif kv is not None:
            assert q is not None, "q should not be None, when kv is not None"
            assert q.device == kv.device, "the devices of q and kv should be same"
            query = q
            key = kv[:, :, 0], kv[:, :, 1]
            device = query.device
        else:
            assert (
                q is not None and k is not None and q is not None
            ), "q, k, v should not be None"
            assert (
                q.device == k.device and k.device == v.device
            ), "the devices of q, k and v should be same"
            query = q
            key, value = k, v
            device = query.device
        
        batch_size, seqlen = query.shape[0], query.shape[1]
        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", query, key * softmax_scale)
        # if key_padding_mask is not None:
        #     padding_mask = torch.full(
        #         (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
        #     )
        #     padding_mask.masked_fill_(key_padding_mask, 0.0)
        #     # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        #     scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=device), 1
            )
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, value)
        return output


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Only supports the padded mode"""
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
        if kv.shape[3] != q.shape[2]:  # MQA/GQA
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # causal mask needs to take into account the difference between seqlen_q and seqlen_k
            row_idx = rearrange(
                torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1"
            )
            col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
            sk = (
                seqlen_k
                if key_padding_mask is None
                else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
            )
            causal_mask = col_idx > row_idx + sk - seqlen_q
            scores = scores.masked_fill(causal_mask, -10000.0)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output
