# Copyright (c) 2024, DeepLink.

import torch
import torch_npu
from einops import rearrange

__all__ = ["ApplyRotaryEmb", "ApplyRotaryEmbQKV_"]


class ApplyRotaryEmb(torch.autograd.Function):
    """
    Apply rotary positional embedding to input tensor x.
    Args:
        x (Tensor): Input tensor x is of shape [seq_length, ... , dim]
        cos (Tensor): Input tensor cos is of shape [seq_length, ..., dim]
        sin (Tensor): Input tensor sin is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
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
        out = torch.empty_like(x)

        re_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin[:seqlen], "s d -> s 1 d")

        cat_cos = torch.cat([re_cos, re_cos], -1)
        cat_sin = torch.cat([re_sin, re_sin], -1)

        rot = torch_npu.npu_rotary_mul(x[..., :rotary_dim], cat_cos, cat_sin)
        out[..., :rotary_dim].copy_(rot)
        if rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])

        ctx.save_for_backward(cat_cos, cat_sin)
        ctx.interleaved = interleaved
        return out

    @staticmethod
    def backward(ctx, do):
        cat_cos, cat_sin = ctx.saved_tensors
        headdim = do.shape[-1]
        rotary_dim = cat_cos.shape[-1]

        dx = torch.empty_like(do)
        dx_rot = torch_npu.npu_rotary_mul(
            do[..., :rotary_dim], cat_cos, torch.neg(cat_sin)
        )
        dx.copy_(dx_rot)

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
        cat_cos = torch.cat([re_cos, re_cos], -1)
        cat_sin = torch.cat([re_sin, re_sin], -1)
        q_out = torch_npu.npu_rotary_mul(q_ro, cat_cos, cat_sin)
        q_ro.copy_(q_out)

        k_ro = (
            qkv[:, 1, :, :rotary_dim]
            if len(qkv.shape) == 4
            else qkv[:, :, 1, :, :rotary_dim]
        )
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
        cat_cos_k = torch.cat([re_cos_k, re_cos_k], -1)
        cat_sin_k = torch.cat([re_sin_k, re_sin_k], -1)
        k_out = torch_npu.npu_rotary_mul(k_ro, cat_cos_k, cat_sin_k)
        k_ro.copy_(k_out)

        ctx.save_for_backward(cat_cos, cat_sin, cat_cos_k, cat_sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cat_cos, cat_sin, cat_cos_k, cat_sin_k = ctx.saved_tensors
        rotary_dim = cat_cos.shape[-1]

        dq_ro = (
            dqkv[:, 0, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 0, :, :rotary_dim]
        )
        dq_out = torch_npu.npu_rotary_mul(dq_ro, cat_cos, torch.neg(cat_sin))
        dq_ro.copy_(dq_out)

        dk_ro = (
            dqkv[:, 1, :, :rotary_dim]
            if len(dqkv.shape) == 4
            else dqkv[:, :, 1, :, :rotary_dim]
        )
        dk_out = torch_npu.npu_rotary_mul(dk_ro, cat_cos_k, torch.neg(cat_sin_k))
        dk_ro.copy_(dk_out)

        return dqkv, None, None, None, None, None
