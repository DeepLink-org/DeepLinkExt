# Copyright (c) 2024, DeepLink.

import torch
import torch.nn as nn
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")

__all__ = ["FlashSelfAttention", "FlashCrossAttention"]


class FlashAttentionQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv=None,
        q=None,
        k=None,
        v=None,
        kv=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
    ):
        # The current default input layout for flash attention is BSND
        input_layout = "BSND"
        device = None
        if qkv is not None:
            query = qkv[:, :, 0]
            key, value = qkv[:, :, 1], qkv[:, :, 2]
            device = qkv.device
        elif kv is not None:
            assert q is not None, "q should not be None, when kv is not None"
            assert q.device == kv.device, "the devices of q and kv should be same"
            query = q
            key, value = kv[:, :, 0], kv[:, :, 1]
            device = kv.device
        else:
            assert (
                q is not None and k is not None and q is not None
            ), "q, k, v should not be None"
            assert (
                q.device == k.device and k.device == v.device
            ), "the devices of q, k and v should be same"
            query = q
            key, value = k, v
            device = q.device
        gen = torch.Generator(device)

        if softmax_scale is None:
            softmax_scale = key.shape[-1] ** (-0.5)

        head_num = query.shape[2]
        out = torch.empty_like(query)
        (
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd(
            out,
            query,
            key,
            value,
            gen,
            dropout_p,
            softmax_scale,
            causal,
            head_num,
            input_layout,
        )

        ctx.save_for_backward(
            qkv,
            q,
            k,
            v,
            kv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            qkv,
            q,
            k,
            v,
            kv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

        if qkv is not None:
            dqkv = torch.empty_like(qkv)
            ext.fa_bwd(
                dqkv[:, :, 0],
                dqkv[:, :, 1],
                dqkv[:, :, 2],
                dout,
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.head_num,
                ctx.input_layout,
            )
            return dqkv, None, None, None, None, None, None, None
        elif kv is not None:
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            ext.fa_bwd(
                dq,
                dkv[:, :, 0],
                dkv[:, :, 1],
                dout,
                q,
                kv[:, :, 0],
                kv[:, :, 1],
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.head_num,
                ctx.input_layout,
            )
            return None, dq, None, None, dkv, None, None, None
        else:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            ext.fa_bwd(
                dq,
                dk,
                dv,
                dout,
                q,
                k,
                v,
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.head_num,
                ctx.input_layout,
            )
            return None, dq, dk, dv, None, None, None, None


class FlashAttentionKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, dropout_p, softmax_scale, causal):
        # The current default input layout for flash attention is BSND
        input_layout = "BSND"
        assert q.device == kv.device, "the devices of q and kv should be same"
        gen = torch.Generator(device=q.device)

        if softmax_scale is None:
            softmax_scale = kv[:, :, 0].shape[-1] ** (-0.5)

        head_num = q.shape[2]
        out = torch.empty_like(q)
        (
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd(
            out,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            gen,
            dropout_p,
            softmax_scale,
            causal,
            head_num,
            input_layout,
        )

        ctx.save_for_backward(
            q,
            kv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            q,
            kv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)

        ext.fa_bwd(
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            dout,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.head_num,
            ctx.input_layout,
        )
        return dq, dkv, None, None, None, None


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
        if cu_seqlens is None:
            # padded
            return FlashAttentionQKVPackedFunc.apply(
                qkv,
                q,
                k,
                v,
                kv,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
            )
        else:
            # unpadded
            raise RuntimeError(
                "DeepLinkSelfAttention does not support the unpadded mode now"
            )


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
        if cu_seqlens is None:
            # padded
            return FlashAttentionKVPackedFunc.apply(
                q,
                kv,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
            )
        else:
            # unpadded
            raise RuntimeError(
                "DeepLinkCrossAttention does not support the unpadded mode now"
            )
