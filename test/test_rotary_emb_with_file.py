import torch
import torch_dipu
from einops import rearrange
import dipu_ext.ext_

# 添加apply文件夹到python path中
from ext_apply_rotary import TorchApplyRotaryEmbQKV_, DeeplLinkApplyRotaryEmbQKV_
    
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
apply_rotary_emb_qkv_ = DeeplLinkApplyRotaryEmbQKV_.apply
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
backward_result_actual = DeeplLinkApplyRotaryEmbQKV_.backward(ctx, dqkv)
assert torch.allclose(backward_result_actual[0], backward_result['dqkv_result'])