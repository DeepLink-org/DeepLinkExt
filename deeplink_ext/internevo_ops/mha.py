# Copyright (c) 2023, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

import torch.nn as nn


__all__ = [
    "MultiHeadAttention",
    "MultiHeadAttentionKVPacked",
    "MultiHeadAttentionQKVPacked",
    "MultiHeadAttentionVarLen",
    "MultiHeadAttentionVarLenKVPacked",
    "MultiHeadAttentionVarLenQKVPacked",
    "SelfAttention",
    "CrossAttention",
]

assert hasattr(ext, "mha_fwd") and hasattr(ext, "mha_bwd")


class MultiHeadAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_fwd(
            q,
            k,
            v,
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, rng.get_state())
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        dq, dk, dv = ext.mha_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            None,
            None,
            None,
        )
        return dq, dk, dv, None, None, None, None


class MultiHeadAttentionKVPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_fwd(
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(q, kv, out, softmax_lse, rng.get_state())
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        q, kv, out, softmax_lse, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        ext.mha_bwd(
            dout,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            out,
            softmax_lse,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
        )
        return dq, dkv, None, None, None, None


class MultiHeadAttentionQKVPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_fwd(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(qkv, out, softmax_lse, rng.get_state())
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        qkv, out, softmax_lse, rng_state = ctx.saved_tensors
        dqkv = torch.empty_like(qkv)
        rng = torch.Generator(device=qkv.device)
        rng.set_state(rng_state)
        ext.mha_bwd(
            dout,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            out,
            softmax_lse,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
        )
        return dqkv, None, None, None, None


assert hasattr(ext, "mha_varlen_fwd") and hasattr(ext, "mha_varlen_bwd")


class MultiHeadAttentionVarLen(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng.get_state()
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = ctx.saved_tensors
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        dq, dk, dv = ext.mha_varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            None,
            None,
            None,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


class MultiHeadAttentionVarLenKVPacked(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_varlen_fwd(
            q,
            kv[:, 0],
            kv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(
            q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng.get_state()
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        (
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = ctx.saved_tensors
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        ext.mha_varlen_bwd(
            dout,
            q,
            kv[:, 0],
            kv[:, 1],
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            dq,
            dkv[:, 0],
            dkv[:, 1],
        )
        return dq, dkv, None, None, None, None, None, None, None, None


class MultiHeadAttentionVarLenQKVPacked(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, rng, S_dmask = ext.mha_varlen_fwd(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(qkv, out, softmax_lse, cu_seqlens, rng.get_state())
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        qkv, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        dqkv = torch.empty_like(qkv)
        rng = torch.Generator(device=qkv.device)
        rng.set_state(rng_state)
        ext.mha_varlen_bwd(
            dout,
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            out,
            softmax_lse,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
        )
        return dqkv, None, None, None, None, None, None, None, None


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
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
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
            return MultiHeadAttentionQKVPacked.apply(
                qkv,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )
        else:
            # unpadded
            return MultiHeadAttentionVarLenQKVPacked.apply(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )


class CrossAttention(nn.Module):
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
            return MultiHeadAttentionKVPacked.apply(
                q,
                kv,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )
        else:
            # unpadded
            return MultiHeadAttentionVarLenKVPacked.apply(
                q,
                kv,
                cu_seqlens,
                cu_seqlens_k,
                max_seqlen,
                max_seqlen_k,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )
