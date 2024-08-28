# Copyright (c) 2024, DeepLink.

import torch
import torch_npu
import torch.nn as nn

__all__ = ["FlashSelfAttention", "FlashCrossAttention"]


class FlashSelfAttention(nn.Module):
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
        self.dropout_p = attention_dropout

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
        if padded:
            if qkv is not None:
                query, key, value = qkv.unbind(dim=2)
            elif kv is not None:
                assert q is not None, "q should not be None, when kv is not None"
                assert q.device == kv.device, "the devices of q and kv should be same"
                query = q
                key, value = kv.unbind(dim=2)
            else:
                assert (
                    q is not None and k is not None and q is not None
                ), "q, k, v should not be None"
                assert (
                    q.device == k.device and k.device == v.device
                ), "the devices of q, k and v should be same"
                query, key, value = q, k, v

            if softmax_scale is None:
                softmax_scale = query.shape[-1] ** (-0.5)
            head_num = query.shape[-2]

            seqlen_q = min(query.shape[1], 2048)
            seqlen_kv = min(key.shape[1], 2048)

            if seqlen_q < 2048:
                sparse_mode = 0
            else:
                sparse_mode = 2

            attention_mask = (
                torch.triu(
                    torch.ones(
                        [seqlen_q, seqlen_kv], dtype=torch.bool, device=query.device
                    ),
                    diagonal=1,
                )
                if causal
                else None
            )

            out = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
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
        else:
            # unpadded
            if qkv is not None:
                query, key, value = qkv.unbind(dim=1)
            elif kv is not None:
                assert q is not None, "q should not be None, when kv is not None"
                assert q.device == kv.device, "the devices of q and kv should be same"
                query = q
                key, value = kv.unbind(dim=1)
            else:
                assert (
                    q is not None and k is not None and q is not None
                ), "q, k, v should not be None"
                assert (
                    q.device == k.device and k.device == v.device
                ), "the devices of q, k and v should be same"
                query, key, value = q, k, v

            cu_seqlens = next(
                (x for x in (cu_seqlens, cu_seqlens_q, cu_seqlens_k) if x is not None),
                None,
            )
            max_seqlen = next(
                (x for x in (max_seqlen, max_seqlen_q, max_seqlen_k) if x is not None),
                None,
            )

            if softmax_scale is None:
                softmax_scale = query.shape[-1] ** (-0.5)
            head_num = query.shape[-2]

            assert (
                cu_seqlens is not None
            ), "cu_seqlens should not be None, when using varlen flash attention"
            cu_seqlens = cu_seqlens[1:].tolist()
            seqlen = min(max_seqlen, 2048)
            attention_mask = (
                torch.triu(
                    torch.ones([seqlen, seqlen], dtype=torch.bool, device=query.device),
                    diagonal=1,
                )
                if causal
                else None
            )

            if max_seqlen < 2048:
                sparse_mode = 0
            else:
                sparse_mode = 2

            out = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                head_num,
                "TND",
                atten_mask=attention_mask,
                scale=softmax_scale,
                pre_tockens=query.shape[0],
                next_tockens=0,
                keep_prob=1 - dropout_p,
                sparse_mode=sparse_mode,
                actual_seq_qlen=cu_seqlens,
                actual_seq_kvlen=cu_seqlens,
            )[0]

            return out


class FlashCrossAttention(nn.Module):

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

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
        padded = all(x is None for x in (cu_seqlens, cu_seqlens_k))
        if padded:
            # padded
            if self.softmax_scale is None:
                self.softmax_scale = q.shape[-1] ** (-0.5)
            k = kv[:, :, 0]
            v = kv[:, :, 1]

            s0 = q.shape[1]
            s1 = kv.shape[1]
            head_num = q.shape[-2]

            if s0 == s1 and s0 < 2048 and s1 < 2048:
                sparse_mode = 0
            else:
                sparse_mode = 2

            seqlen_q = min(s0, 2048)
            seqlen_k = min(s1, 2048)

            attention_mask = (
                torch.triu(
                    torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=q.device),
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
                scale=self.softmax_scale,
                keep_prob=1 - self.dropout_p,
                pre_tockens=seqlen_k,
                next_tockens=0,
                sparse_mode=sparse_mode,
            )[0]

            return out

        else:
            # unpadded
            if self.softmax_scale is None:
                self.softmax_scale = q.shape[-1] ** (-0.5)
            k = kv[:, 0]
            v = kv[:, 1]
            n = q.shape[1]
            cu_seqlens_q = cu_seqlens[1:].tolist()
            cu_seqlens_k = cu_seqlens_k[1:].tolist()
            seqlen_q = min(max_seqlen, 2048)
            seqlen_k = min(max_seqlen_k, 2048)

            if max_seqlen > 2048:
                sparse_mode = 2
            else:
                sparse_mode = 0

            attention_mask = (
                torch.triu(
                    torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=q.device),
                    diagonal=1,
                )
                if causal
                else None
            )
            out = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                n,
                "TND",
                atten_mask=attention_mask,
                scale=self.softmax_scale,
                pre_tockens=q.shape[0],  # seq_len
                next_tockens=0,  # 0
                keep_prob=1 - self.dropout_p,
                sparse_mode=sparse_mode,
                actual_seq_qlen=cu_seqlens_q,
                actual_seq_kvlen=cu_seqlens_k,
            )[0]
            return out
