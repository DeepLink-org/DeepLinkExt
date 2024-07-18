import numbers
import torch
from torch.nn import init

import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "gmm")


__all__ = ["gmm", "GroupedGemmFunc"]


def gmm_forward(a, b, batchSizes, transA, transB):
    assert not (transA and transB), "'transA' and 'transB' can't both be true"
    assert batchSizes.ndim == 1, "Expected 1d tensor for batchSizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if transA else 3)

    shape = (
        (batchSizes.shape[0], a.shape[1], b.shape[1])
        if transA
        else (a.shape[0], (b.shape[1] if transB else b.shape[2]))
    )
    out = torch.empty(*shape, device=a.device, dtype=a.dtype)

    if batchSizes.is_cuda:
        ext.gmm(out, a, b, batchSizes, transA, transB)
    else:
        ext.gmm(out, a, b, batchSizes.cuda, transA, transB)

    return out


class GroupedGemm(torch.autograd.Fuction):
    @staticmethod
    def forward(ctx, a, b, batchSizes, transB):
        ctx.save_for_backward(a, b, batchSizes)
        ctx.transB = transB
        return gmm_forward(a, b, batchSizes, False, transB)
    
    @staticmethod
    def backward(ctx, grad):
        a, b, batchSizes = ctx.saved_tensors
        transB = ctx.transB
        
        gradA = gmm_forward(grad, b, batchSizes, False, transB)
        
        lhs, rhs = (grad, a) if transB else (a, grad)
        gradB = gmm_forward(lhs, rhs, batchSizes, True, False)
        
        return gradA, gradB, None, None