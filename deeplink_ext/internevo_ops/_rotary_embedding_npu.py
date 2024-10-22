# Copyright (c) 2024, DeepLink.

import torch
import torch_npu
from einops import rearrange, repeat
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

__all__ = ["ApplyRotaryEmb"]


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L35
class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        in_place: bool = False,
    ):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        *_, seqlen, _, head_dim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2

        assert rotary_dim <= head_dim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)

        if interleaved:
            cos = repeat(cos[:seqlen].unsqueeze(0).unsqueeze(2), "... d -> ... (d 2)")
            sin = repeat(sin[:seqlen].unsqueeze(0).unsqueeze(2), "... d -> ... (d 2)")
        else:
            # "s d -> 1 s 1 d"
            cos = cos[:seqlen].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
            sin = sin[:seqlen].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.in_place = in_place
        if interleaved:
            x_ro = x[..., :rotary_dim]
            out_ro = npu_rotary_position_embedding(x_ro, cos, sin, 1)
            if in_place:
                x[..., :rotary_dim].copy_(out_ro)
                return x
            if rotary_dim < head_dim:
                out = torch.empty_like(x)
                out[..., :rotary_dim].copy_(out_ro)
                out[..., rotary_dim:].copy_(x[..., rotary_dim:])
                return out
            return out_ro
        else:
            x_ro = x[..., :rotary_dim]
            out_ro = torch_npu.npu_rotary_mul(x_ro, cos, sin)
            if in_place:
                x[..., :rotary_dim].copy_(out_ro)
                return x
            if rotary_dim < head_dim:
                out = torch.empty_like(x)
                out[..., :rotary_dim].copy_(out_ro)
                out[..., rotary_dim:].copy_(x[..., rotary_dim:])
                return out
            return out_ro

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        rotary_dim = cos.shape[-1]
        head_dim = grad_out.shape[-1]
        if ctx.interleaved:
            grad_out_ro = grad_out[..., :rotary_dim]
            grad_input_ro = npu_rotary_position_embedding(
                grad_out_ro, cos, torch.neg(sin), 1
            )
            if ctx.in_place:
                grad_out[..., :rotary_dim].copy_(grad_input_ro)
                return grad_out, None, None, None, None
            if rotary_dim < head_dim:
                grad_input = torch.empty_like(grad_out)
                grad_input[..., :rotary_dim].copy_(grad_input_ro)
                grad_input[..., rotary_dim:].copy_(grad_out[..., rotary_dim:])
                return grad_input, None, None, None, None
            return grad_input_ro, None, None, None, None
        else:
            grad_out_ro = grad_out[..., :rotary_dim]
            grad_input_ro = torch_npu.npu_rotary_mul(grad_out_ro, cos, torch.neg(sin))
            if ctx.in_place:
                grad_out[..., :rotary_dim].copy_(grad_input_ro)
                return grad_out, None, None, None, None
            if rotary_dim < head_dim:
                grad_input = torch.empty_like(grad_out)
                grad_input[..., :rotary_dim].copy_(grad_input_ro)
                grad_input[..., rotary_dim:].copy_(grad_out[..., rotary_dim:])
                return grad_input, None, None, None, None
            return grad_input_ro, None, None, None, None
