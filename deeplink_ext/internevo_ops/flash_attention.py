# Copyright (c) 2024, DeepLink.

import torch
import torch_dipu
import torch.nn as nn
import deeplink_ext.cpp_extensions as ext


if torch_dipu.dipu.vendor_type == "NPU":
    assert hasattr(ext, "custom_fa_fwd") and hasattr(ext, "custom_fa_bwd")
    assert hasattr(ext, "custom_fa_varlen_fwd") and hasattr(ext, "custom_fa_varlen_bwd")
else:
    assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")
    assert hasattr(ext, "fa_varlen_fwd") and hasattr(ext, "fa_varlen_bwd")

__all__ = ["FlashSelfAttention", "FlashCrossAttention"]


class CustomizedFlashAttentionQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        q,
        k,
        v,
        kv,
        dropout_p,
        softmax_scale,
        causal,
    ):
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

        device = query.device
        gen = torch.Generator(device)

        if softmax_scale is None:
            softmax_scale = key.shape[-1] ** (-0.5)

        seqlen_q = min(query.shape[1], 2048)
        seqlen_kv = min(key.shape[1], 2048)
        attention_mask = torch.triu(torch.ones([seqlen_q, seqlen_kv], dtype=torch.bool, device=device), diagonal=1) if causal else None
        out = torch.empty_like(query)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_fwd(
            out,
            gen,
            query,
            key,
            value,
            None,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            -1,
            -1,
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
        ctx.causal = causal
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
            ext.custom_fa_bwd(
                dqkv[:, :, 0],
                dqkv[:, :, 1],
                dqkv[:, :, 2],
                dout,
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                None,
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
            )
            return dqkv, None, None, None, None, None, None, None
        elif kv is not None:
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            ext.custom_fa_bwd(
                dq,
                dkv[:, :, 0],
                dkv[:, :, 1],
                dout,
                q,
                kv[:, :, 0],
                kv[:, :, 1],
                None,
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
            )
            return None, dq, None, None, dkv, None, None, None
        else:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            ext.custom_fa_bwd(
                dq,
                dk,
                dv,
                dout,
                q,
                k,
                v,
                None,
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
            )
            return None, dq, dk, dv, None, None, None, None


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

        device = query.device
        gen = torch.Generator(device)

        if softmax_scale is None:
            softmax_scale = key.shape[-1] ** (-0.5)

        batch_size = query.shape[0]
        seqlen_q = query.shape[1]
        head_num = query.shape[2]
        out = torch.empty_like(query)
        softmax_lse = torch.empty(
            [batch_size, head_num, seqlen_q], dtype=torch.float32, device=device
        )

        ext.fa_fwd(
            out,
            softmax_lse,
            gen,
            query,
            key,
            value,
            None,
            dropout_p,
            softmax_scale,
            causal,
            -1,
            -1,
        )
        ctx.save_for_backward(
            qkv,
            q,
            k,
            v,
            kv,
            out,
            softmax_lse,
        )
        ctx.gen = gen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
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
            softmax_lse,
        ) = ctx.saved_tensors

        if qkv is not None:
            dqkv = torch.empty_like(qkv)
            ext.fa_bwd(
                dqkv[:, :, 0],
                dqkv[:, :, 1],
                dqkv[:, :, 2],
                dout,
                ctx.gen,
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                None,
                out,
                softmax_lse,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
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
                ctx.gen,
                q,
                kv[:, :, 0],
                kv[:, :, 1],
                None,
                out,
                softmax_lse,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
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
                ctx.gen,
                q,
                k,
                v,
                None,
                out,
                softmax_lse,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
            )
            return None, dq, dk, dv, None, None, None, None


class CustomizedFlashAttentionVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        q,
        k,
        v,
        kv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
    ):
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

        device = query.device
        gen = torch.Generator(device)

        if softmax_scale is None:
            softmax_scale = key.shape[-1] ** (-0.5)

        assert (
            cu_seqlens is not None
        ), "cu_seqlens should not be None, when using varlen flash attention"
        cu_seqlens = cu_seqlens[1:].tolist()
        seqlen = min(max_seqlen, 2048)
        attention_mask = torch.triu(torch.ones([seqlen, seqlen], dtype=torch.bool, device=device), diagonal=1) if causal else None

        out = torch.empty_like(query)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_varlen_fwd(
            out,
            gen,
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            None,
            attention_mask,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            -1,
            -1,
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
        ctx.causal = causal
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
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
            ext.custom_fa_varlen_bwd(
                dqkv[:, 0],
                dqkv[:, 1],
                dqkv[:, 2],
                dout,
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
                ctx.cu_seqlens,
                ctx.cu_seqlens,
                None,
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.max_seqlen,
                ctx.max_seqlen,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
            )
            return dqkv, None, None, None, None, None, None, None, None, None
        elif kv is not None:
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            ext.custom_fa_varlen_bwd(
                dq,
                dkv[:, 0],
                dkv[:, 1],
                dout,
                q,
                kv[:, 0],
                kv[:, 1],
                ctx.cu_seqlens,
                ctx.cu_seqlens,
                None,
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.max_seqlen,
                ctx.max_seqlen,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
            )
            return None, dq, None, None, dkv, None, None, None, None, None
        else:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            ext.custom_fa_varlen_bwd(
                dq,
                dk,
                dv,
                dout,
                q,
                k,
                v,
                ctx.cu_seqlens,
                ctx.cu_seqlens,
                None,
                out,
                attention_mask,
                dropout_mask,
                softmax_max,
                softmax_sum,
                softmax_out,
                ctx.max_seqlen,
                ctx.max_seqlen,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                -1,
                -1,
            )
            return None, dq, dk, dv, None, None, None, None, None, None


class CustomizedFlashAttentionKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, dropout_p, softmax_scale, causal):
        assert q.device == kv.device, "the devices of q and kv should be same"
        gen = torch.Generator(device=q.device)

        if softmax_scale is None:
            softmax_scale = kv.shape[-1] ** (-0.5)

        seqlen_q = min(q.shape[1], 2048)
        seqlen_kv = min(kv.shape[1], 2048)
        attention_mask = torch.triu(torch.ones([seqlen_q, seqlen_kv], dtype=torch.bool, device=q.device), diagonal=1) if causal else None
        out = torch.empty_like(q)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_fwd(
            out,
            gen,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            None,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            -1,
            -1,
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
        ctx.causal = causal
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

        ext.custom_fa_bwd(
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            dout,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            None,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            -1,
            -1,
        )
        return dq, dkv, None, None, None, None


class FlashAttentionKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, dropout_p, softmax_scale, causal):
        # The current default input layout for flash attention is BSND
        assert q.device == kv.device, "the devices of q and kv should be same"
        gen = torch.Generator(device=q.device)

        if softmax_scale is None:
            softmax_scale = kv.shape[-1] ** (-0.5)

        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        head_num = q.shape[2]
        out = torch.empty_like(q)
        softmax_lse = torch.empty(
            [batch_size, head_num, seqlen_q], dtype=torch.float32, device=q.device
        )

        ext.fa_fwd_v3(
            out,
            softmax_lse,
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            gen,
            dropout_p,
            softmax_scale,
            causal,
        )

        ctx.save_for_backward(
            q,
            kv,
            out,
            softmax_lse,
        )
        ctx.gen = gen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        (
            q,
            kv,
            out,
            softmax_lse,
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
            ctx.gen,
            softmax_lse,
            ctx.causal,
            ctx.dropout_p,
            ctx.softmax_scale,
        )
        return dq, dkv, None, None, None, None


class CustomizedFlashAttentionVarlenKVPackedFunc(torch.autograd.Function):
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
    ):
        # The current default input layout for varlen flash attention is TND
        assert q.device == kv.device, "the devices of q and kv should be same"
        gen = torch.Generator(device=q.device)

        if softmax_scale is None:
            softmax_scale = kv.shape[-1] ** (-0.5)

        assert (
            cu_seqlens_q is not None and cu_seqlens_k is not None
        ), "cu_seqlens_q and cu_seqlens_k should not be None, when using varlen flash attention"
        cu_seqlens_q = cu_seqlens_q[1:].tolist()
        cu_seqlens_k = cu_seqlens_k[1:].tolist()

        out = torch.empty_like(q)
        (
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_varlen_fwd(
            out,
            q,
            kv[:, 0],
            kv[:, 1],
            gen,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
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
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
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

        ext.fa_varlen_bwd(
            dq,
            dkv[:, 0],
            dkv[:, 1],
            dout,
            q,
            kv[:, 0],
            kv[:, 1],
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
        )
        return dq, dkv, None, None, None, None, None, None, None


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
            # padded
            if torch_dipu.dipu.vendor_type == "NPU":
                return CustomizedFlashAttentionQKVPackedFunc.apply(
                    qkv,
                    q,
                    k,
                    v,
                    kv,
                    dropout_p,
                    softmax_scale,
                    causal if causal is not None else self.causal,
                )

            else:
                return FlashAttentionQKVPackedFunc.apply(
                    qkv,
                    q,
                    k,
                    v,
                    kv,
                    dropout_p,
                    softmax_scale,
                    causal if causal is not None else self.causal,
                )
        else:
            # unpadded
            cu_seqlens = next(
                (x for x in (cu_seqlens, cu_seqlens_q, cu_seqlens_k) if x is not None),
                None,
            )
            max_seqlen = next(
                (x for x in (max_seqlen, max_seqlen_q, max_seqlen_k) if x is not None),
                None,
            )
            return FlashAttentionVarlenQKVPackedFunc.apply(
                qkv,
                q,
                k,
                v,
                kv,
                cu_seqlens,
                max_seqlen,
                dropout_p,
                softmax_scale,
                causal if causal is not None else self.causal,
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
        padded = all(x is None for x in (cu_seqlens, cu_seqlens_k))
        if padded:
            # padded
            if torch_dipu.dipu.vendor_type == "NPU":
                return CustomizedFlashAttentionKVPackedFunc.apply(
                    q,
                    kv,
                    self.dropout_p if self.training else 0.0,
                    self.softmax_scale,
                    causal if causal is not None else self.causal,
                )
            else:
                return FlashAttentionKVPackedFunc.apply(
                    q,
                    kv,
                    self.dropout_p if self.training else 0.0,
                    self.softmax_scale,
                    causal if causal is not None else self.causal,
                )
        else:
            # unpadded
            return CustomizedFlashAttentionVarlenKVPackedFunc.apply(
                q,
                kv,
                cu_seqlens,
                cu_seqlens_k,
                max_seqlen,
                max_seqlen_k,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
            )
