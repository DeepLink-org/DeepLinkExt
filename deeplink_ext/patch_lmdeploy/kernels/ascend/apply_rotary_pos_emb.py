import torch
import deeplink_ext.cpp_extensions as ext
from torch import Tensor


def apply_rotary_pos_emb(query_states: Tensor, key_states: Tensor, cos: Tensor, sin: Tensor,
                         position_ids: Tensor, position_ids_1d: Tensor, q_embed=None, k_embed=None):
    bs, head, dim = query_states.shape
    numKeyValueHeads = key_states.shape[1]
    seqlens = [(min(position_id), max(position_id) + 1) for position_id in position_ids.tolist()]
    query_states = query_states.reshape(bs, head*dim)
    key_states = key_states.reshape(bs, numKeyValueHeads*dim)
    cos = torch.cat([cos[i:j] for i, j in seqlens]).view(1, bs, 1, -1)
    sin = torch.cat([sin[i:j] for i, j in seqlens]).view(1, bs, 1, -1)
    ext.rotary_embedding_v2(query_states, key_states, cos, sin, dim)
    return query_states.view(bs, head, dim), key_states.view(bs, numKeyValueHeads, dim)
