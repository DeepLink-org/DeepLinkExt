import deeplink_ext.cpp_extensions as ext

import torch
import torch_dipu

print(ext)
print("^.^ " * 20)
ext.customed_op(torch.rand(3).cuda(), torch.rand(4).cuda())
print("^.^ " * 20)
