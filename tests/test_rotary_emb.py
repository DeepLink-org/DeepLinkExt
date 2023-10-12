import torch
import torch_dipu
from einops import rearrange


import dipu_ext.ext_

def torch_apply_rotary(x1, x2, cos, sin, conj):
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
    out1 = out1.to(torch.float16)
    out2 = out2.to(torch.float16)
    return out1, out2

class ApplyRotaryEmbQKV_(torch.autograd.Function):
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
        # q1, q2 = q_ro.chunk(2, dim=-1) if not interleaved else (q_ro[..., ::2], q_ro[..., 1::2])
        # q1, q2 = torch_apply_rotary(q1, q2, rearrange(cos[:seqlen], 's d -> s 1 d'),
        #                         rearrange(sin[:seqlen], 's d -> s 1 d'), False)
        # qkv[:, :, 0, :, :rotary_dim] = torch.cat((q1,q2),dim=-1)
        dipu_ext.ext_.apply_rotary(q_ro, q_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), False)

        k_ro = qkv[:, :, 1, :, :rotary_dim]
        # k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
        # k1, k2 = torch_apply_rotary(k1, k2, rearrange(cos[:seqlen], 's d -> s 1 d'),
        #                         rearrange(sin[:seqlen], 's d -> s 1 d'), False)
        # qkv[:, :, 1, :, :rotary_dim] = torch.cat((k1,k2),dim=-1)
        dipu_ext.ext_.apply_rotary(k_ro, k_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), False)


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
        # dq1, dq2 = (dq_ro.chunk(2, dim=-1) if not ctx.interleaved
        #             else (dq_ro[..., ::2], dq_ro[..., 1::2]))
        # dq1, dq2 = torch_apply_rotary(dq1, dq2, rearrange(cos[:seqlen], 's d -> s 1 d'),
        #                         rearrange(sin[:seqlen], 's d -> s 1 d'), True)
        # dqkv[:, :, 0, :, :rotary_dim] = torch.cat((dq1, dq2), dim=-1)
        dipu_ext.ext_.apply_rotary(dq_ro, dq_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), True)
        
        
        dk_ro = dqkv[:, :, 1, :, :rotary_dim]
        # dk1, dk2 = (dk_ro.chunk(2, dim=-1) if not ctx.interleaved
        #             else (dk_ro[..., ::2], dk_ro[..., 1::2]))
        # dk1,dk2 = torch_apply_rotary(dk1, dk2, rearrange(cos[:seqlen], 's d -> s 1 d'),
        #                         rearrange(sin[:seqlen], 's d -> s 1 d'), True)
        # dqkv[:, :, 1, :, :rotary_dim] = torch.cat((dk1, dk2), dim=-1)
        dipu_ext.ext_.apply_rotary(dk_ro, dk_ro,  rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), True)
        
        return dqkv, None, None, None, None, None
    
# 测试 数据从internLM的pytorch版本生成
import pickle
# 从文件中加载保存的参数和结果数据
with open('forward_params.pkl', 'rb') as file:
    forward_params = pickle.load(file)

with open('forward_result.pkl', 'rb') as file:
    forward_result = pickle.load(file)

with open('backward_paras.pkl', 'rb') as file:
    backward_params = pickle.load(file)

with open('backward_result.pkl', 'rb') as file:
    backward_result = pickle.load(file)

# 前向forward测试
apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply
forward_result_actual = apply_rotary_emb_qkv_(forward_params['qkv'], forward_params['cos'], forward_params['sin'], forward_params['cos_k'], forward_params['sin_k'])
assert torch.allclose(forward_result_actual, forward_result['qkv_result'])


# 反向backward测试
class VirtualContext:
    def __init__(self, saved_tensors, interleaved):
        self.saved_tensors = saved_tensors
        self.interleaved = interleaved
dqkv = backward_params['dqkv']
cos = backward_params['cos']
sin = backward_params['sin']
cos_k = backward_params['cos_k']
sin_k = backward_params['sin_k']
interleaved = backward_params['interleaved']
ctx = VirtualContext([cos, sin, cos_k, sin_k], interleaved)
backward_result_actual = ApplyRotaryEmbQKV_.backward(ctx, dqkv)
assert torch.allclose(backward_result_actual[0], backward_result['dqkv_result'])