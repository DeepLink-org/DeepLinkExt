# Copyright (c) 2023, DeepLink.

import torch
import torch.nn as nn
import einops

__all__ = ["SelfAttention", "CrossAttention"]


class SelfAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None):
        """Only supports padded"""
        seqlen = qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or q.shape[-1] ** -0.5
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class CrossAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None):
        """Only supports padded"""
        seqlen_q = q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        if kv.shape[3] != q.shape[2]:
            kv = einops.repeat(
                kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3]
            )
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or q.shape[-1] ** -0.5
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if causal:
            row_idx = einops.rearrange(
                torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1"
            )
            col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
            sk = seqlen_k
            causal_mask = col_idx > row_idx + sk - seqlen_q
            scores = scores.masked_fill(causal_mask, -10000.0)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output
