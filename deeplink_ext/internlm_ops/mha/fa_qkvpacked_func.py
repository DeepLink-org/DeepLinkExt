# Copyright (c) 2023, DeepLink.

import torch
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "fa_fwd") and hasattr(ext, "fa_bwd")


class DeepLinkFlashAttentionQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv=None, q=None, k=None, v=None, kv=None, dropout_p=0.0, softmax_scale=None, causal=False):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        head_num = 0
        if qkv is not None:
            assert (q, k, v, kv) == (None, None, None, None)
            q = qkv[:, 0]
            k = qkv[:, 1]
            v = qkv[:, 2]
            head_num = qkv.shape[2]
        else:
            assert q is not None
            if kv is not None:
                k = kv[:, 0]
                v = kv[:, 1]
            else:
                assert k is not None and v is not None
            head_num = q.shape[1]
        (
            out,
            attention_mask,
            dropout_mask,
            softmax_max,
            softmax_sum,
            softmax_out,
        ) = ext.fa_fwd(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            head_num,
        )
        ctx.save_for_backward(
            qkv, q, k, v, kv,
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
            qkv, q, k, v, kv,
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
                dkv[:, 0],
                dkv[:, 1],
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
