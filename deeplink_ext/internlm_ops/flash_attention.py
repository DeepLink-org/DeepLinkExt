# Copyright (c) 2024, DeepLink.

import torch
import torch.nn as nn
import torch_dipu
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")


class DeepLinkFlashAttentionQKVPackedFunc(torch.autograd.Function):
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
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        head_num = 0
        if qkv is not None:
            assert (q, k, v, kv) == (None, None, None, None)
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            v = qkv[:, :, 2]
            head_num = qkv.shape[3]
        else:
            assert q is not None
            if kv is not None:
                k = kv[:, :, 0]
                v = kv[:, :, 1]
            else:
                assert k is not None and v is not None
            head_num = q.shape[2]
        # The current default input layout for flash attention is BSND
        input_layout = "BSND"
        out = torch.empty_like(q)
        gen = torch_dipu._C._create_dipu_generator(-1)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd_v2(
            out,
            q,
            k,
            v,
            gen,
            attention_mask,
            dropout_p,
            softmax_scale,
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
        attention_mask = (
            torch.Tensor().cuda() if attention_mask is None else attention_mask
        )
        dropout_mask = torch.Tensor().cuda() if dropout_mask is None else dropout_mask
        if qkv is not None:
            dqkv = torch.empty_like(qkv)
            ext.fa_bwd(
                dqkv[:, :, 0],
                dqkv[:, :, 1],
                dqkv[:, :, 2],
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
            )
            return None, dq, dk, dv, None, None, None, None


class DeepLinkFlashAttentionKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, dropout_p, softmax_scale, causal):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_num = q.shape[2]
        (
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd(
            q, kv[:, :, 0], kv[:, :, 1], dropout_p, softmax_scale, causal, head_num
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
        attention_mask = (
            torch.Tensor().cuda() if attention_mask is None else attention_mask
        )
        dropout_mask = torch.Tensor().cuda() if dropout_mask is None else dropout_mask
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
        )
        return dq, dkv, None, None, None, None


class DeepLinkSelfAttention(nn.Module):
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
        if causal:
            assert causal == self.causal
        if dropout_p:
            assert dropout_p == self.dropout_p
        if softmax_scale:
            assert softmax_scale == self.softmax_scale
        if cu_seqlens is None:
            # padded
            return DeepLinkFlashAttentionQKVPackedFunc.apply(
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


class DeepLinkCrossAttention(nn.Module):
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
            return DeepLinkFlashAttentionKVPackedFunc.apply(
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
