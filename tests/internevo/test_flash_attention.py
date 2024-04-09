import torch

from deeplink_ext.internevo_ops.flash_attention import FlashSelfAttention, FlashCrossAttention
from deeplink_ext.internevo_ops.flash_attention_fallback import SelfAttention, CrossAttention


def test_self_attention():
    batch = 8
    seqlen = 32
    nheads = 16
    headdim = 64
    
    q_ref = torch.rand([batch, seqlen, nheads, headdim], requires_grad=True)
    k_ref = torch.rand([batch, seqlen , nheads, headdim], requires_grad=True)
    v_ref = torch.rand([batch, seqlen , nheads, headdim], requires_grad=True)
    qkv_ref = torch.stack([q_ref, k_ref, v_ref], 2)
    q_ext = q_ref.clone().detach().to(torch.float16).cuda().requires_grad_()
    k_ext = k_ref.clone().detach().to(torch.float16).cuda().requires_grad_()
    v_ext = v_ref.clone().detach().to(torch.float16).cuda().requires_grad_()
    
    model_ref = SelfAttention()
    model_ext = FlashSelfAttention()
    out_ref = model_ref(qkv_ref)
    out = model_ext(None, q_ext, k_ext, v_ext, None)
    out_ref.backward(torch.ones_like(out_ref))
    out.backward(torch.ones_like(out))
    
    assert torch.allclose(out.cpu(), out_ref.to(torch.float16), rtol=1e-3, atol=1e-3)
    assert torch.allclose(q_ext.grad.cpu(), q_ref.grad.to(torch.float16), rtol=1e-3, atol=1e-3)
    assert torch.allclose(k_ext.grad.cpu(), k_ref.grad.to(torch.float16), rtol=1e-3, atol=1e-3)
    assert torch.allclose(v_ext.grad.cpu(), v_ref.grad.to(torch.float16), rtol=1e-3, atol=1e-3)


def test_cross_attention():
    batch = 8
    seqlen = 32
    nheads = 16
    headdim = 64
    
    q_ref = torch.rand([batch, seqlen, nheads, headdim], requires_grad=True)
    kv_ref = torch.rand([batch, seqlen , 2, nheads, headdim], requires_grad=True)
    q_ext = q_ref.clone().detach().to(torch.float16).cuda().requires_grad_()
    kv_ext = kv_ref.clone().detach().to(torch.float16).cuda().requires_grad_()
    
    model_ref = CrossAttention()
    model_ext = FlashCrossAttention()
    out_ref = model_ref(q_ref, kv_ref)
    out = model_ext(q_ext, kv_ext)
    out_ref.backward(torch.ones_like(out_ref))
    out.backward(torch.ones_like(out))
    
    assert torch.allclose(out.cpu(), out_ref.to(torch.float16), rtol=1e-3, atol=1e-3)
    assert torch.allclose(q_ext.grad.cpu(), q_ref.grad.to(torch.float16), rtol=1e-3, atol=1e-3)
    assert torch.allclose(kv_ext.grad.cpu(), kv_ref.grad.to(torch.float16), rtol=1e-3, atol=1e-3)
    