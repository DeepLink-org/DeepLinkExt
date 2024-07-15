# Copyright (c) 2024, DeepLink.

import torch
import torch_dipu
import deeplink_ext.cpp_extensions as ext

if torch_dipu.dipu.vendor_type == "NPU":
    assert hasattr(ext, "custom_fa_fwd") and hasattr(ext, "custom_fa_bwd")
    assert hasattr(ext, "custom_fa_varlen_fwd") and hasattr(ext, "custom_fa_varlen_bwd")
else:
    assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")
    assert hasattr(ext, "fa_varlen_fwd") and hasattr(ext, "fa_varlen_bwd")

__all__ = [
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_func",
]


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        device = qkv.device
        gen = torch.Generator(device)

        seqlen_qkv = min(qkv.shape[1], 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_qkv, seqlen_qkv], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        head_num = qkv.shape[-2]
        input_layout = "BSND"
        out = torch.empty_like(qkv[:, :, 0])
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_fwd(
            out,
            gen,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            alibi_slopes,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            head_num,
            input_layout,
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
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            qkv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

        dqkv = torch.empty_like(qkv)
        ext.custom_fa_bwd(
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            dout,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            ctx.alibi_slopes,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.head_num,
            ctx.input_layout,
        )
        return dqkv, None, None, None, None, None, None, None


class CustomizedFlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        device = qkv.device
        gen = torch.Generator(device)

        seqlen_qkv = min(qkv.shape[1], 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_qkv, seqlen_qkv], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        head_num = qkv.shape[-2]
        input_layout = "BSND"
        out = torch.empty_like(qkv[:, :, 0])
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_fwd(
            out,
            gen,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            alibi_slopes,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            head_num,
            input_layout,
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
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            qkv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

        dqkv = torch.empty_like(qkv)
        ext.custom_fa_bwd(
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            dout,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            ctx.alibi_slopes,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.head_num,
            ctx.input_layout,
        )
        return dqkv, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if torch_dipu.dipu.vendor_type == "NPU":
        return CustomizedFlashAttnQKVPackedFunc.apply(
            qkv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )
    else:
        return FlashAttnQKVPackedFunc.apply(
            qkv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        seqlen_q = min(q.shape[1], 2048)
        seqlen_kv = min(kv.shape[1], 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_kv], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        head_num = q.shape[-2]
        input_layout = "BSND"
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
            alibi_slopes,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
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
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout, *args):
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
            ctx.alibi_slopes,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.head_num,
            ctx.input_layout,
        )
        return dq, dkv, None, None, None, None, None, None, None


class CustomizedFlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        seqlen_q = min(q.shape[1], 2048)
        seqlen_kv = min(kv.shape[1], 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_kv], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        head_num = q.shape[-2]
        input_layout = "BSND"
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
            alibi_slopes,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
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
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout, *args):
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
            ctx.alibi_slopes,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.head_num,
            ctx.input_layout,
        )
        return dq, dkv, None, None, None, None, None, None, None


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if torch_dipu.dipu.vendor_type == "NPU":
        return CustomizedFlashAttnKVPackedFunc.apply(
            q,
            kv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )
    else:
        return FlashAttnKVPackedFunc.apply(
            q,
            kv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        seqlen_q = min(q.shape[1], 2048)
        seqlen_k = min(k.shape[1], 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        head_num = q.shape[-2]
        input_layout = "BSND"
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
            k,
            v,
            alibi_slopes,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            head_num,
            input_layout,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
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
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

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
            ctx.alibi_slopes,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.head_num,
            ctx.input_layout,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


class CustomizedFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        seqlen_q = min(q.shape[1], 2048)
        seqlen_k = min(k.shape[1], 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        head_num = q.shape[-2]
        input_layout = "BSND"
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
            k,
            v,
            alibi_slopes,
            attention_mask,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            head_num,
            input_layout,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
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
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

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
            ctx.alibi_slopes,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.head_num,
            ctx.input_layout,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if torch_dipu.dipu.vendor_type == "NPU":
        return CustomizedFlashAttnFunc.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )
    else:
        return FlashAttnFunc.apply(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        device = qkv.device
        gen = torch.Generator(device)

        cu_seqlens = cu_seqlens[1:].tolist()
        seqlen = min(max_seqlen, 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen, seqlen], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        out = torch.empty_like(qkv[:, 0])
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_varlen_fwd(
            out,
            gen,
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            alibi_slopes,
            attention_mask,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
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
        ctx.causal = causal
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            qkv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

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
            ctx.alibi_slopes,
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
            ctx.window_size[0],
            ctx.window_size[1],
        )
        return dqkv, None, None, None, None, None, None, None, None, None


class CustomizedFlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        device = qkv.device
        gen = torch.Generator(device)

        cu_seqlens = cu_seqlens[1:].tolist()
        seqlen = min(max_seqlen, 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen, seqlen], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        out = torch.empty_like(qkv[:, 0])
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_varlen_fwd(
            out,
            gen,
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            alibi_slopes,
            attention_mask,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
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
        ctx.causal = causal
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            qkv,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

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
            ctx.alibi_slopes,
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
            ctx.window_size[0],
            ctx.window_size[1],
        )
        return dqkv, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if torch_dipu.dipu.vendor_type == "NPU":
        return CustomizedFlashAttnVarlenQKVPackedFunc.apply(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )
    else:
        return FlashAttnVarlenQKVPackedFunc.apply(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
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
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        cu_seqlens_q = cu_seqlens_q[1:].tolist()
        cu_seqlens_k = cu_seqlens_k[1:].tolist()
        seqlen_q = min(max_seqlen_q, 2048)
        seqlen_k = min(max_seqlen_k, 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        out = torch.empty_like(q)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_varlen_fwd(
            out,
            gen,
            q,
            kv[:, 0],
            kv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            alibi_slopes,
            attention_mask,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
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
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        return out

    @staticmethod
    def backward(ctx, dout, *args):
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
        ext.custom_fa_varlen_bwd(
            dq,
            dkv[:, 0],
            dkv[:, 1],
            dout,
            q,
            kv[:, 0],
            kv[:, 1],
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.alibi_slopes,
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
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
        )
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None


class CustomizedFlashAttnVarlenKVPackedFunc(torch.autograd.Function):
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
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        cu_seqlens_q = cu_seqlens_q[1:].tolist()
        cu_seqlens_k = cu_seqlens_k[1:].tolist()
        seqlen_q = min(max_seqlen_q, 2048)
        seqlen_k = min(max_seqlen_k, 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        out = torch.empty_like(q)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_varlen_fwd(
            out,
            gen,
            q,
            kv[:, 0],
            kv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            alibi_slopes,
            attention_mask,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
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
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        return out

    @staticmethod
    def backward(ctx, dout, *args):
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
        ext.custom_fa_varlen_bwd(
            dq,
            dkv[:, 0],
            dkv[:, 1],
            dout,
            q,
            kv[:, 0],
            kv[:, 1],
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.alibi_slopes,
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
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
        )
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    if torch_dipu.dipu.vendor_type == "NPU":
        return CustomizedFlashAttnVarlenKVPackedFunc.apply(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )
    else:
        return FlashAttnVarlenKVPackedFunc.apply(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )


class FlashAttnVarlenFunc(torch.autograd.Function):
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
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        cu_seqlens_q = cu_seqlens_q[1:].tolist()
        cu_seqlens_k = cu_seqlens_k[1:].tolist()
        seqlen_q = min(max_seqlen_q, 2048)
        seqlen_k = min(max_seqlen_k, 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        out = torch.empty_like(q)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_varlen_fwd(
            out,
            gen,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            alibi_slopes,
            attention_mask,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
        )

        ctx.save_for_backward(
            q,
            k,
            v,
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
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

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
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.alibi_slopes,
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
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CustomizedFlashAttnVarlenFunc(torch.autograd.Function):
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
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        device = q.device
        gen = torch.Generator(device)

        cu_seqlens_q = cu_seqlens_q[1:].tolist()
        cu_seqlens_k = cu_seqlens_k[1:].tolist()
        seqlen_q = min(max_seqlen_q, 2048)
        seqlen_k = min(max_seqlen_k, 2048)
        attention_mask = (
            torch.triu(
                torch.ones([seqlen_q, seqlen_k], dtype=torch.bool, device=device),
                diagonal=1,
            )
            if causal
            else None
        )

        out = torch.empty_like(q)
        (
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.custom_fa_varlen_fwd(
            out,
            gen,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            alibi_slopes,
            attention_mask,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
        )

        ctx.save_for_backward(
            q,
            k,
            v,
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
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ctx.saved_tensors

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
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.alibi_slopes,
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
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    if torch_dipu.dipu.vendor_type == "NPU":
        return CustomizedFlashAttnVarlenFunc.apply(
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
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            block_table,
        )
    else:
        return FlashAttnVarlenFunc.apply(
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
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            block_table,
        )
