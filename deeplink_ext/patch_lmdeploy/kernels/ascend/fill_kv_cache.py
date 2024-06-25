import torch
import deeplink_ext.cpp_extensions as ext
from torch import Tensor
from ...engine.devices.ascend import ASCENDDeviceUtils



def fill_kv_cache(key_states: Tensor, value_states: Tensor, key_caches: Tensor,
                  value_caches: Tensor, q_start_loc: Tensor, q_seq_length: Tensor,
                  kv_seq_length: Tensor, max_q_seq_length: int,
                  block_offsets: Tensor):
    """fill key/value state to cache for paged attention."""
    assert hasattr(ASCENDDeviceUtils.step_context, "kv_start_indices")
    kv_start_indices = ASCENDDeviceUtils.step_context.kv_start_indices
    ASCENDDeviceUtils.add_kv_states(key_states, value_states)
    dest_index_copy_kv(key_states, kv_start_indices, key_caches)
    dest_index_copy_kv(value_states, kv_start_indices, value_caches)


def dest_index_copy_kv(states, dest_loc, caches):
    block_num, block_size, head, dim = caches.size()
    caches_tmp = caches.view(block_num * block_size, head, dim)
    ext.dest_index_copy_kv(states, dest_loc, caches_tmp)
    caches[:] = caches_tmp.view(block_num, block_size, head, dim)
