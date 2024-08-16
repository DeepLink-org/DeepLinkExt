
# Copyright (c) 2024, DeepLink.

import os
import torch

def func_for_in_out(func, pth):
    cnt = 0
    pth = os.path.join(pth, f"{func.__name__}")
    os.makedirs(pth, exist_ok=True)

    def wrapper(*args, **kwargs):
        nonlocal cnt
        pth_inout = os.path.join(pth, f"inputs_outputs_{cnt}.pt")
        inputs = (args, kwargs)

        outputs = func(*args, **kwargs)

        torch.save({"inputs": inputs, "outputs": outputs}, pth_inout)
        cnt += 1
        return outputs

    return wrapper


# for testing

# def f1(*args, **kwargs):
#     print(args, kwargs)

# f1 = func_for_in_out(f1, "")

# f1(1,2,3,a=11,b=22,c=33)
# f1(1,2,3,a=11,b=22,c=33)
