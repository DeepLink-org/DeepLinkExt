# Copyright (c) 2024, DeepLink.

import torch
from typing import Callable, Any

__all__ = [
    "calculate_fwd_and_bwd",
    "call_module",
    "call_autograd_func",
    "copy_to_cpu",
    "allclose",
]


def calculate_fwd_and_bwd(func: Callable[..., Any], *args: tuple, **kwargs: dict):
    output_forward = func(*args, **kwargs)
    grads = []
    if torch.is_tensor(output_forward):
        output_forward.backward(torch.ones_like(output_forward))
    elif isinstance(output_forward, (list, tuple)):
        assert torch.is_tensor(output_forward[0]), "output_forward[0] is not a tensor"
        output_forward[0].backward(torch.ones_like(output_forward[0]))
    else:
        raise RuntimeError(
            "the result of forward is not a tensor or list or tuple of tensor"
        )
    for arg in args:
        if torch.is_tensor(arg) and arg.requires_grad:
            grads.append(arg.grad)
    return output_forward, grads


def call_module(module: torch.nn.Module, *args: tuple, **kwargs: dict):
    return calculate_fwd_and_bwd(module, *args, **kwargs)


def call_autograd_func(
    autograd_func: torch.autograd.Function, device, dtype, *args: tuple, **kwargs: dict
):
    class Module(torch.nn.Module):
        def __init__(self, func):
            super(Module, self).__init__()
            self.func = func

        def forward(self, *args):
            return self.func.apply(*args)

    return call_module(Module(autograd_func).to(device).to(dtype), *args, **kwargs)


def copy_to_cpu(tensors: list[torch.Tensor], dtype=None):
    if dtype is None:
        dtype = torch.float32
    return [
        tensor.detach().clone().to(dtype).cpu().requires_grad_(tensor.requires_grad)
        for tensor in tensors
    ]


def allclose(expected_vals: list, real_vals: list, rtol=1e-05, atol=1e-08):
    assert len(expected_vals) == len(real_vals), "length of outputs is not same"
    for i in range(len(expected_vals)):
        assert type(expected_vals[i]) == type(
            real_vals[i]
        ), "the type of expected_vals[{index}] is {type1}, but real_vals[{index}] is {type2}.".format(
            index=i, type1=type(expected_vals[i]), type2=type(real_vals[i])
        )
        if isinstance(expected_vals[i], torch.Tensor):
            assert isinstance(real_vals[i], torch.Tensor)
            return torch.allclose(
                expected_vals[i].cpu().to(torch.float32),
                real_vals[i].cpu().to(torch.float32),
                rtol,
                atol,
            )
        elif isinstance(expected_vals[i], (tuple, list)):
            assert isinstance(real_vals[i], (tuple, list))
            allclose(expected_vals[i], real_vals[i], rtol, atol)
        elif isinstance(expected_vals[i], dict):
            assert isinstance(real_vals[i], dict)
            for key, val in expected_vals[i].items():
                assert key in real_vals.keys(), "key {k} not in real_val.keys()".format(
                    k=key
                )
                allclose(val, real_vals[key], rtol, atol)
        # Primitive type
        else:
            return abs(real_vals[i] - expected_vals[i]) <= atol + rtol * abs(
                expected_vals[i]
            )
