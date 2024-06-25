import torch
import math
import deeplink_ext.cpp_extensions as ext
from torch import Tensor
from ...engine.devices.ascend import ASCENDDeviceUtils


mask_cache = {}


def paged_attention_fwd(query_states: Tensor, key_cache: Tensor, value_cache: Tensor,
                        attn_output: Tensor, block_offsets: Tensor, q_start_loc: Tensor,
                        q_seqlens: Tensor, kv_seqlens: Tensor, max_seqlen: int):
    is_decoding = query_states.shape[-3] == q_seqlens.size(0)
    block_num, block_size, head, dim = key_cache.size()
    kv_cache_len = block_num * block_size
    k = key_cache.reshape(block_num * block_size, head, dim)
    v = value_cache.reshape(block_num * block_size, head, dim)
    if not is_decoding:
        key_states = ASCENDDeviceUtils.key_states
        value_states = ASCENDDeviceUtils.value_states
        flash_context_attention(query_states, key_states, value_states, attn_output, k,
                                v, block_offsets.to(torch.int32), q_start_loc, q_seqlens.tolist(),
                                kv_seqlens.tolist(), block_size, kv_cache_len)
    else:
        paged_token_attention(query_states, k, v, attn_output, kv_seqlens.tolist(),
                              block_offsets.to(torch.int32), block_size)


def flash_context_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: list,
    kv_seqlens: list,
    block_size: int,
    kv_cache_len: int,
):
    batch, head, dim = q_start_loc.shape[0], query_states.shape[1], query_states.shape[2]
    numKeyValueHeads = value_states.shape[1]
    assert key_states.shape[1] == value_states.shape[1]
    for i in range(batch):
        start = q_start_loc[i]
        end = start + q_seqlens[i]
        single_seqlen = int(end - start)
        single_q = query_states[start:end].view(1, single_seqlen, -1)
        single_k = key_states[start:end].reshape(1, single_seqlen, -1)
        single_v = value_states[start:end].reshape(1, single_seqlen, -1)
        single_out = attn_output[start:end, :].view(1, single_seqlen, -1)
        if q_seqlens[i] == kv_seqlens[i]:
            if single_seqlen not in mask_cache:
                mask = torch.tril(torch.ones(single_seqlen, single_seqlen, dtype=torch.bool), diagonal=0).cuda()
                mask = torch.logical_not(mask)
                mask_cache[single_seqlen] = mask
                print(f"cache mask in context attention, seqLen:{single_seqlen}")
            mask = mask_cache[single_seqlen]
            ext.prompt_flash_attention(single_out, single_q, single_k, single_v,
                                    mask, [kv_seqlens[i]], kv_seqlens[i], head, numKeyValueHeads, dim)
        else:
            key_cache = key_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
            value_cache = value_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
            for j in range(q_seqlens[i]):
                single_q = query_states[start+j:start+j+1].view(1, 1, -1)
                single_out = attn_output[start+j:start+j+1].view(1, 1, -1)
                if f"{q_seqlens[i]}_{kv_seqlens[i]}" not in mask_cache:
                    mask = torch.tril(torch.ones(q_seqlens[i], kv_seqlens[i], dtype=torch.bool), diagonal=kv_seqlens[i] - q_seqlens[i]).cuda()
                    mask = torch.logical_not(mask)
                    mask_cache[f"{q_seqlens[i]}_{kv_seqlens[i]}"] = mask
                    print(f"cache mask in context attention, seqLen:{q_seqlens[i]}_{kv_seqlens[i]}")
                mask = mask_cache[f"{q_seqlens[i]}_{kv_seqlens[i]}"]
                ext.paged_attention(single_out, single_q, key_cache, value_cache, mask[j:j+1],
                                    [kv_seqlens[i]],  head, numKeyValueHeads,
                                    dim, block_offsets[i:i+1], block_size
                                    )


def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seqlens, block_table:torch.Tensor, block_size):
    numKeyValueHeads = k_cache.shape[1]
    assert k_cache.shape[1] == v_cache.shape[1]
    bs, head, dim = q.shape
    kv_cache_len = k_cache.shape[0]
    q = q.reshape(bs, 1, head*dim)
    k_cache = k_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
    v_cache = v_cache.reshape(1, kv_cache_len, numKeyValueHeads*dim)
    ext.paged_attention(attn_output.view(q.shape), q, k_cache, v_cache, None,
                        kv_seqlens, head, numKeyValueHeads,
                        dim, block_table, block_size
                        )
