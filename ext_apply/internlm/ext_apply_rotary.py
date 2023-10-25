import torch
import dipu_ext.ext_
from einops import rearrange


# Rotary_emb
# torch绕过实现函数
def torch_apply_rotary(x1, x2, cos, sin, conj):
    data_dtype = x1.dtype
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    out1 = out1.to(data_dtype)
    out2 = out2.to(data_dtype)
    return out1, out2

class TorchApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        """
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q_ro = qkv[:, :, 0, :, :rotary_dim]
        q1, q2 = q_ro.chunk(2, dim=-1) if not interleaved else (q_ro[..., ::2], q_ro[..., 1::2])
        q1, q2 = torch_apply_rotary(q1, q2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), False)
        qkv[:, :, 0, :, :rotary_dim] = torch.cat((q1,q2),dim=-1)
        k_ro = qkv[:, :, 1, :, :rotary_dim]
        k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
        k1, k2 = torch_apply_rotary(k1, k2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), False)
        qkv[:, :, 1, :, :rotary_dim] = torch.cat((k1,k2),dim=-1)
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv
    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        _, seqlen, _, _, headdim = dqkv.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq_ro = dqkv[:, :, 0, :, :rotary_dim]
        dq1, dq2 = (dq_ro.chunk(2, dim=-1) if not ctx.interleaved
                    else (dq_ro[..., ::2], dq_ro[..., 1::2]))
        dq1, dq2 = torch_apply_rotary(dq1, dq2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), True)
        dqkv[:, :, 0, :, :rotary_dim] = torch.cat((dq1, dq2), dim=-1)
        dk_ro = dqkv[:, :, 1, :, :rotary_dim]
        dk1, dk2 = (dk_ro.chunk(2, dim=-1) if not ctx.interleaved
                    else (dk_ro[..., ::2], dk_ro[..., 1::2]))
        dk1,dk2 = torch_apply_rotary(dk1, dk2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), True)
        dqkv[:, :, 1, :, :rotary_dim] = torch.cat((dk1, dk2), dim=-1)
        return dqkv, None, None, None, None, None
try:
    import dipu_ext.ext_
    print("USE ext apply_rotary")
    class DeeplLinkApplyRotaryEmbQKV_(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            rotary_seqlen, rotary_dim = cos.shape
            rotary_dim *= 2
            assert rotary_dim <= headdim
            assert seqlen <= rotary_seqlen
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
            q_ro = qkv[:, :, 0, :, :rotary_dim]
            dipu_ext.ext_.apply_rotary(q_ro, q_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), False, interleaved)
            k_ro = qkv[:, :, 1, :, :rotary_dim]
            dipu_ext.ext_.apply_rotary(k_ro, k_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), False, interleaved)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.interleaved = interleaved
            return qkv
        @staticmethod
        def backward(ctx, dqkv):
            cos, sin, cos_k, sin_k = ctx.saved_tensors
            interleaved = ctx.interleaved
            _, seqlen, _, _, headdim = dqkv.shape
            rotary_dim = cos.shape[-1]
            rotary_dim *= 2
            dq_ro = dqkv[:, :, 0, :, :rotary_dim]
            dipu_ext.ext_.apply_rotary(dq_ro, dq_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), True, interleaved)
            dk_ro = dqkv[:, :, 1, :, :rotary_dim]
            dipu_ext.ext_.apply_rotary(dk_ro, dk_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), True, interleaved)
            return dqkv, None, None, None, None, None
except:
    print("NOT USING ext apply_rotary")
    pass
