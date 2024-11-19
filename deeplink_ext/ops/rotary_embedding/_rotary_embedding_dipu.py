# Copyright (c) 2024, DeepLink.
# Copyright (c) 2024, InternEvo.

import torch
from einops import rearrange
import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "apply_rotary")
from deeplink_ext.cpp_extensions import apply_rotary

__all__ = ["ApplyRotaryEmb", "ApplyRotaryEmbQKV_","apply_rotary"]


class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        _, seqlen, _, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = x_ro.chunk(2, dim=-1)
        out = torch.empty_like(x)
        out_ro = out[..., :rotary_dim]
        o1, o2 = out_ro.chunk(2, dim=-1)
        re_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin[:seqlen], "s d -> s 1 d")
        apply_rotary(
            x1,
            x2,
            re_cos,
            re_sin,
            o1,
            o2,
            False,
        )
        if rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(re_cos, re_sin)
        ctx.interleaved = interleaved
        return out

    @staticmethod
    def backward(ctx, do):
        re_cos, re_sin = ctx.saved_tensors
        headdim = do.shape[-1]
        rotary_dim = re_cos.shape[-1]
        rotary_dim *= 2
        do_ro = do[..., :rotary_dim]
        do1, do2 = do_ro.chunk(2, dim=-1)
        dx = torch.empty_like(do)
        dx_ro = dx[..., :rotary_dim]
        dx1, dx2 = dx_ro.chunk(2, dim=-1)

        apply_rotary(
            do1,
            do2,
            re_cos,
            re_sin,
            dx1,
            dx2,
            True,
        )
        if rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        """
            qkv: (total, 3, nheads, headdim) / (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        # len(qkv.shape) == 4 means the format of qkv is (total, 3, nheads, headdim) which is packed,
        # otherwise the format of qkv is (batch_size, seqlen, 3, nheads, headdim) which is unpacked.
        # We handle both packed qkv and unpacked qkv scenario in this class.
        three = qkv.shape[1] if len(qkv.shape) == 4 else qkv.shape[2]
        assert three == 3
        seqlen = None if len(qkv.shape) == 4 else qkv.shape[1]
        rotary_seqlen, rotary_dim = cos.shape
        if len(qkv.shape) != 4:
            assert seqlen <= rotary_seqlen
        headdim = qkv.shape[-1]
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert (
            sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        )

        q_ro = (
            qkv[:, 0, :, :rotary_dim]
            if len(qkv.shape) == 4
            else qkv[:, :, 0, :, :rotary_dim]
        )
        q1, q2 = q_ro.chunk(2, dim=-1)
        re_cos = (
            rearrange(cos, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(cos[:seqlen], "s d -> s 1 d")
        )
        re_sin = (
            rearrange(sin, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(sin[:seqlen], "s d -> s 1 d")
        )
        apply_rotary(
            q1,
            q2,
            re_cos,
            re_sin,
            q1,
            q2,
            False
        )

        k_ro = (
            qkv[:, 1, :, :rotary_dim]
            if len(qkv.shape) == 4
            else qkv[:, :, 1, :, :rotary_dim]
        )
        k1, k2 = k_ro.chunk(2, dim=-1)
        re_cos_k = (
            rearrange(cos_k, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(cos_k[:seqlen], "s d -> s 1 d")
        )
        re_sin_k = (
            rearrange(sin_k, "s d -> s 1 d")
            if len(qkv.shape) == 4
            else rearrange(sin_k[:seqlen], "s d -> s 1 d")
        )
        apply_rotary(
            k1,
            k2,
            re_cos_k,
            re_sin_k,
            k1,
            k2,
            False
        )

        ctx.save_for_backward(re_cos, re_sin, re_cos_k, re_sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        re_cos, re_sin, re_cos_k, re_sin_k = ctx.saved_tensors
        rotary_dim = re_cos.shape[-1]
        rotary_dim *= 2

        dq_ro = (
            dqkv[:, 0, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 0, :, :rotary_dim]
        )
        dq1, dq2 = dq_ro.chunk(2, dim=-1)
        _torch_apply_rotary_func(dq1, dq2, re_cos, re_sin, dq1, dq2, True)
        apply_rotary(
            dq1,
            dq2,
            re_cos,
            re_sin,
            dq1,
            dq2,
            True
        )

        dk_ro = (
            dqkv[:, 1, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 1, :, :rotary_dim]
        )
        dk1, dk2 = dk_ro.chunk(2, dim=-1)
        apply_rotary(
            dk1,
            dk2,
            re_cos_k,
            re_sin_k,
            dk1,
            dk2,
            True
        )
        return dqkv, None, None, None, None, None
