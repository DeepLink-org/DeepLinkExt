# Copyright (c) 2024, DeepLink.
# Copyright (c) 2024, InternEvo.
# Copyright (c) 2023, Tri Dao.

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


__all__ = ["SelfAttention", "CrossAttention"]


def multi_head_attention_func(
    q, k, v, softmax_scale, drop, causal=None, key_padding_mask=None
):
    batch_size, seqlen = q.shape[0], q.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if key_padding_mask is not None:
        padding_mask = torch.full(
            (batch_size, seqlen),
            float("-inf"),
            dtype=scores.dtype,
            device=scores.device,
        )
        padding_mask.masked_fill_(key_padding_mask, 0.0)
        scores.add_(rearrange(padding_mask, "b s -> b 1 1 s"))
    if causal:
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen),
                float("-inf"),
                dtype=scores.dtype,
                device=scores.device,
            ),
            1,
        )
        scores.add_(causal_mask)
    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    attention_drop = drop(attention)
    return torch.einsum("bhts,bshd->bthd", attention_drop, v)


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
        padded = all(x is None for x in (cu_seqlens, cu_seqlens_q, cu_seqlens_k))
        drop = nn.Dropout(dropout_p)
        if padded:
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

            causal = self.causal if causal is None else causal
            softmax_scale = (
                softmax_scale
                if softmax_scale is not None
                else 1.0 / math.sqrt(query.shape[-1])
            )
            output = multi_head_attention_func(
                query,
                key,
                value,
                softmax_scale,
                drop,
                causal,
            )
            return output
        else:
            # unpadded
            if qkv is not None:
                query, key, value = qkv.unbind(dim=1)
            elif kv is not None:
                assert q is not None, "q should not be None, when kv is not None"
                assert q.device == kv.device, "the devices of q and kv should be same"
                # adapt to GQA
                if kv.shape[2] != q.shape[1]:  # MQA/GQA
                    kv = repeat(
                        kv, "... hkv d -> ... (hkv g) d", g=q.shape[1] // kv.shape[2]
                    )
                query = q
                key, value = kv.unbind(dim=1)
            else:
                assert (
                    q is not None and k is not None and q is not None
                ), "q, k, v should not be None"
                assert (
                    q.device == k.device and k.device == v.device
                ), "the devices of q, k and v should be same"
                # adapt to GQA
                if k.shape[1] != q.shape[1] and v.shape[1] != q.shape[1]:  # MQA/GQA
                    k = repeat(
                        k, "... hkv d -> ... (hkv g) d", g=q.shape[1] // k.shape[1]
                    )
                    v = repeat(
                        v, "... hkv d -> ... (hkv g) d", g=q.shape[1] // v.shape[1]
                    )
                query = q
                key, value = k, v

            cu_seqlens = next(
                (x for x in (cu_seqlens, cu_seqlens_q, cu_seqlens_k) if x is not None),
                None,
            )
            max_seqlen = next(
                (x for x in (max_seqlen, max_seqlen_q, max_seqlen_k) if x is not None),
                None,
            )

            batch_size = len(cu_seqlens) - 1
            _, head_num, head_dim = query.size()
            device = query.device
            causal = self.causal if causal is None else causal
            softmax_scale = (
                softmax_scale
                if softmax_scale is not None
                else 1.0 / math.sqrt(query.shape[-1])
            )

            padded_shape = (batch_size, max_seqlen, head_num, head_dim)
            query_padded = torch.zeros(padded_shape, dtype=query.dtype, device=device)
            key_padded = torch.zeros(padded_shape, dtype=key.dtype, device=device)
            value_padded = torch.zeros(padded_shape, dtype=value.dtype, device=device)

            # Initialize the key_padding_mask as a Boolean mask with False values
            key_padding_mask = torch.zeros(
                (batch_size, max_seqlen), dtype=torch.bool, device=device
            )
            # Fill the key_padding_mask with True values at positions with actual data (cu_seqlens)
            for i in range(batch_size):
                start_idx = cu_seqlens[i]
                end_idx = cu_seqlens[i + 1]
                actual_seq_len = end_idx - start_idx
                key_padding_mask[i, :actual_seq_len] = True
                query_padded[i, :actual_seq_len, :, :] = query[start_idx:end_idx, :, :]
                key_padded[i, :actual_seq_len, :, :] = key[start_idx:end_idx, :, :]
                value_padded[i, :actual_seq_len, :, :] = value[start_idx:end_idx, :, :]

            qkv_padded_result = multi_head_attention_func(
                query_padded,
                key_padded,
                value_padded,
                softmax_scale,
                drop,
                causal,
                key_padding_mask,
            )
            output = torch.zeros_like(query)

            for i in range(batch_size):
                start_idx = cu_seqlens[i]
                end_idx = cu_seqlens[i + 1]
                actual_seq_len = end_idx - start_idx
                output[start_idx:end_idx, :, :] = qkv_padded_result[
                    i, :actual_seq_len, :, :
                ]
            return output


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
        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        padded = all(x is None for x in (cu_seqlens, cu_seqlens_k))
        if padded:
            # padded
            assert kv.shape[0] == q.shape[0] and kv.shape[4] == q.shape[3]
            if kv.shape[3] != q.shape[2]:  # MQA/GQA
                kv = repeat(
                    kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3]
                )
            k, v = kv.unbind(dim=2)

            output = multi_head_attention_func(
                q,
                k,
                v,
                softmax_scale,
                self.drop,
                causal,
            )
            return output
        else:
            # unpadded
            if kv.shape[2] != q.shape[1]:  # MQA/GQA
                kv = repeat(
                    kv, "... hkv d -> ... (hkv g) d", g=q.shape[1] // kv.shape[2]
                )
            k, v = kv.unbind(dim=1)

            batch_size = len(cu_seqlens) - 1
            _, head_num, head_dim = q.size()
            device = q.device

            padded_shape = (batch_size, max_seqlen, head_num, head_dim)
            query_padded = torch.zeros(padded_shape, dtype=q.dtype, device=device)
            key_padded = torch.zeros(padded_shape, dtype=k.dtype, device=device)
            value_padded = torch.zeros(padded_shape, dtype=v.dtype, device=device)

            # Initialize the key_padding_mask as a Boolean mask with False values
            key_padding_mask = torch.zeros(
                (batch_size, max_seqlen), dtype=torch.bool, device=device
            )
            # Fill the key_padding_mask with True values at positions with actual data (cu_seqlens)
            for i in range(batch_size):
                start_idx = cu_seqlens[i]
                end_idx = cu_seqlens[i + 1]
                actual_seq_len = end_idx - start_idx
                key_padding_mask[i, :actual_seq_len] = True
                query_padded[i, :actual_seq_len, :, :] = q[start_idx:end_idx, :, :]
                key_padded[i, :actual_seq_len, :, :] = k[start_idx:end_idx, :, :]
                value_padded[i, :actual_seq_len, :, :] = v[start_idx:end_idx, :, :]

            qkv_padded_result = multi_head_attention_func(
                query_padded,
                key_padded,
                value_padded,
                softmax_scale,
                self.drop,
                causal,
                key_padding_mask,
            )
            output = torch.zeros_like(q)

            for i in range(batch_size):
                start_idx = cu_seqlens[i]
                end_idx = cu_seqlens[i + 1]
                actual_seq_len = end_idx - start_idx
                output[start_idx:end_idx, :, :] = qkv_padded_result[
                    i, :actual_seq_len, :, :
                ]
            return output
