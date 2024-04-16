# Copyright (c) 2024, DeepLink.
# Copyright (c) 2024, InternEvo.
# Copyright (c) 2023, Tri Dao.

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


__all__ = ["SelfAttention", "CrossAttention"]


class SelfAttention(nn.Module):
    """Performs self-attention with support for both padded and unpadded sequences.

    Args:
        causal (bool, optional): If True, applies causal self-attention, meaning each
            position can only attend to previous positions. Default is False.
        softmax_scale (float, optional): Scaling factor applied to the softmax
            operation. If not provided, will be D^{-0.5}. Default is None.
        dropout_p (float, optional): Dropout probability applied to the attention
            scores. Default is 0.0.
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
        dropout_p=0.0,
    ):
        """Performs self-attention on the input sequences.

        Args:
            qkv (torch.Tensor): Input tensor representing queries, keys, and values
                concatenated together. (B, S, 3, H, D) for padded; (total, 3, H, D)
                for unpadded.
            causal (bool, optional): If provided, overrides the class-level 'causal'
                argument for this forward pass. Default is None.
            cu_seqlens (torch.Tensor((batch_size + 1,), dtype=torch.int32), optional):
                Sequence lengths tensor for unpadded sequences. If provided, performs
                attention on unpadded sequences. Default is None.
            max_seqlen (int, optional): Maximum sequence length for unpadded sequences.
                If provided, defines the maximum length of the sequences. Default is
                None.

        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        if cu_seqlens is None:
            # padded
            if qkv is not None:
                query, key, value = qkv.unbind(dim=2)
            elif kv is not None:
                assert q is not None, "q should not be None, when kv is not None"
                assert q.device == kv.device, "the devices of q and kv should be same"
                # adapt to GQA
                if kv.shape[3] != q.shape[2]:  # MQA/GQA
                    kv = repeat(
                        kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3]
                    )
                query = q
                key, value = kv.unbind(dim=2)
            else:
                assert (
                    q is not None and k is not None and q is not None
                ), "q, k, v should not be None"
                assert (
                    q.device == k.device and k.device == v.device
                ), "the devices of q, k and v should be same"
                # adapt to GQA
                if k.shape[2] != q.shape[2] and v.shape[2] != q.shape[2]:  # MQA/GQA
                    k = repeat(
                        k, "... hkv d -> ... (hkv g) d", g=q.shape[2] // k.shape[2]
                    )
                    v = repeat(
                        v, "... hkv d -> ... (hkv g) d", g=q.shape[2] // v.shape[2]
                    )
                query = q
                key, value = k, v
            device = query.device
            seqlen = query.shape[1]
            causal = self.causal if causal is None else causal
            softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
            scores = torch.einsum("bthd,bshd->bhts", query, key * softmax_scale)
            if causal:
                causal_mask = torch.triu(
                    torch.full((seqlen, seqlen), -10000.0, device=device), 1
                )
                scores = scores + causal_mask.to(dtype=scores.dtype)
            attention = torch.softmax(scores, dim=-1, dtype=value.dtype)
            attention_drop = self.drop(attention)
            output = torch.einsum("bhts,bshd->bthd", attention_drop, value)
            return output
        else:
            # unpadded
            raise RuntimeError("SelfAttention does not support the unpadded mode now")


class CrossAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        q,
        kv,
        causal=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        if cu_seqlens is None:
            # padded
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            causal = self.causal if causal is None else causal
            seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            if kv.shape[3] != q.shape[2]:  # MQA/GQA
                kv = repeat(
                    kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3]
                )
            k, v = kv.unbind(dim=2)
            softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
            if causal:
                row_idx = rearrange(
                    torch.arange(seqlen_q, device=q.device, dtype=torch.long),
                    "s -> s 1",
                )
                col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
                causal_mask = col_idx > row_idx
                scores = scores.masked_fill(causal_mask, -10000.0)
            attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
            attention_drop = self.drop(attention)
            output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
            return output
        else:
            # unpadded
            raise RuntimeError("CrossAttention does not support the unpadded mode now")
