import torch
import torch.nn as nn
import torch_dipu
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")

__all__ = ["CrossAttention", "SelfAttention", "ScaledMaskedSoftmax"]


def fa_fwd_out(out, gen, q, k, v, p_dropout, softmax_scale, is_causal, head_num):
    attention_mask, dropout_mask, softmax_max, softmax_sum, softmax_out = ext.fa_fwd(
        out,
        gen,
        q,
        k,
        v,
        p_dropout,
        softmax_scale,
        is_causal,
        head_num,
    )
    return [out, attention_mask, dropout_mask, softmax_max, softmax_sum, softmax_out]


def fa_fwd(q, k, v, p_dropout, softmax_scale, is_causal, head_num):
    out = torch.empty_like(q)
    gen = torch_dipu._C._create_dipu_generator(-1)

    return fa_fwd_out(out, gen, q, k, v, p_dropout, softmax_scale, is_causal, head_num)


def fa_bwd_out(
    grad_q,
    grad_k,
    grad_v,
    grad_out,
    q,
    k,
    v,
    out,
    attention_mask,
    dropout_mask,
    softmax_max,
    softmax_sum,
    softmax_out,
    p_dropout,
    softmax_scale,
    head_num,
):
    ext.fa_bwd(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        q,
        k,
        v,
        out,
        attention_mask,
        dropout_mask,
        softmax_max,
        softmax_sum,
        softmax_out,
        p_dropout,
        softmax_scale,
        head_num,
    )

    return [grad_q, grad_k, grad_v]


def fa_bwd(
    grad_out,
    q,
    k,
    v,
    out,
    attention_mask,
    dropout_mask,
    softmax_max,
    softmax_sum,
    softmax_out,
    p_dropout,
    softmax_scale,
    head_num,
):
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)

    return fa_bwd_out(
        grad_q,
        grad_k,
        grad_v,
        grad_out,
        q,
        k,
        v,
        out,
        attention_mask,
        dropout_mask,
        softmax_max,
        softmax_sum,
        softmax_out,
        p_dropout,
        softmax_scale,
        head_num,
    )


class FlashAttentionKVPackedFunc(torch.autograd.Function):
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
        ) = fa_fwd(
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
        fa_bwd(
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


class FlashAttentionQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, dropout_p, softmax_scale, causal):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        head_num = qkv.shape[3]
        (
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            dropout_p,
            softmax_scale,
            causal,
            head_num,
        )
        ctx.save_for_backward(
            qkv,
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
        )
        return dqkv, None, None, None, None


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
            return FlashAttentionQKVPackedFunc.apply(
                qkv,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
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
            return FlashAttentionKVPackedFunc.apply(
                q,
                kv,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
            )


class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, scale, fixed_triu_mask):
        out = ext.scaled_masked_softmax_fwd(input, mask, scale, fixed_triu_mask)
        ctx.save_for_backward(out, mask)
        ctx.scale = scale
        ctx.fixed_triu_mask = fixed_triu_mask
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, mask = ctx.saved_tensors
        grad_input = ext.scaled_masked_softmax_bwd(
            grad_output, out, mask, ctx.scale, ctx.fixed_triu_mask
        )
        return grad_input, None, None, None
