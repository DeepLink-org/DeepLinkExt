import torch
import torch_dipu
import deeplink_ext
torch.ops.load_library(deeplink_ext.__path__[0] + "/cpp_extensions.cpython-39-x86_64-linux-gnu.so")
print(f"torch.ops.loaded_libraries:{torch.ops.loaded_libraries}")

#print(torch.ops.deeplink_ext_.dest_index_copy_kv)

def code_to_profile():
    x = torch.randn(3,4)
    y = torch.ops.deeplink_ext_.example(x)
    y = torch.ops.deeplink_ext_.example(x.cuda())


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    code_to_profile()
print(p.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))

