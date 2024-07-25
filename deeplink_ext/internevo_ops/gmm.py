import numbers
import torch
from torch.nn import init

import deeplink_ext.cpp_extensions as ext

assert hasattr(ext, "gmm")


__all__ = ["gmm_forward", "GroupedGemm"]


def gmm_forward(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b), "'trans_a' and 'trans_b' can't both be true"
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a
        else (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    out = torch.empty(*shape, device=a.device, dtype=a.dtype)

    if batch_sizes.is_cuda:
        ext.gmm(out, a, b, batch_sizes, trans_a, trans_b)
    else:
        ext.gmm(out, a, b, batch_sizes.cuda(), trans_a, trans_b)

    return out


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return gmm_forward(a, b, batch_sizes, False, trans_b)

    @staticmethod
    def backward(ctx, grad):
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        # gradA = gmm_forward(grad, b, batch_sizes, False, not trans_b)
        # lhs, rhs = (grad, a) if trans_b else (a, grad)
        # gradB = gmm_forward(lhs, rhs, batch_sizes, True, False)

        gradA = torch.empty_like(a)
        gradB = torch.empty_like(b)
        ext.gmm_backward(gradA, gradB, a, b, batch_sizes.cuda(), grad, False, trans_b)


        return gradA, gradB, None, None
