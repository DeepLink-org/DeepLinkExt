# Copyright (c) 2024, DeepLink.

import torch
import torch_dipu


def _patch_vllm():
    import os
    from functools import lru_cache
    from vllm._C import ops as vllm_ops
    from vllm._C import cache_ops as vllm_cache_ops
    from vllm.attention.ops import paged_attn as vllm_paged_attn
    from vllm.attention import selector as vllm_attn_selector
    from vllm.attention.backends import xformers as vllm_attn_xformers
    from vllm.distributed.device_communicators import pynccl_utils as vllm_pynccl_utils
    import deeplink_ext.cpp_extensions as ext

    DEFAULT_PATCH_LIST = [
        "which_attn_to_use",
        "pynccl_utils",
        "rms_norm",
        "rotary_emb",
        "fused_add_rms_norm",
        "silu_and_mul",
        "reshape_and_cache",
        "run_memory_efficient_xformers_forward",
        "page_attn",
    ]
    PATCH_LIST_ENV_NAME = "DEEPLINKEXT_VLLM_PATCH_LIST"
    patch_list_env = os.environ.get(PATCH_LIST_ENV_NAME)
    use_custom_patch_list = patch_list_env is not None
    patch_list = (
        patch_list_env.split(",") if use_custom_patch_list else DEFAULT_PATCH_LIST
    )
    if use_custom_patch_list:
        print(f"[deeplink_ext] use custom vllm patch list: {patch_list}\n", end="")

    def try_patch(op: str):

        def patch_which_attn_to_use():
            def which_attn_to_use(dtype):
                return vllm_attn_selector._Backend.XFORMERS

            vllm_attn_selector._which_attn_to_use = which_attn_to_use

        def patch_pynccl_utils():
            def init_process_group(group=None):
                return

            vllm_pynccl_utils.init_process_group = init_process_group

        def patch_rms_norm():
            def rms_norm(output, input, weight, eps):
                inv_rms_shape = list(input.shape[:-1]) + [1]
                inv_rms = torch.empty(
                    inv_rms_shape, dtype=torch.float32, device=input.device
                )
                ext.rms_norm(output, inv_rms, input, weight.shape, weight, None, eps)
                return

            vllm_ops.rms_norm = rms_norm

        def patch_fused_add_rms_norm():
            def fused_add_rms_norm(input, residual, weight, eps):
                inv_rms_shape = list(input.shape[:-1]) + [1]
                inv_rms = torch.empty(
                    inv_rms_shape, dtype=torch.float32, device=input.device
                )
                # inplace input by add_rms_norm
                ext.add_rms_norm(input, inv_rms, residual, input, residual, weight, eps)
                return

            vllm_ops.fused_add_rms_norm = fused_add_rms_norm

        def patch_rotary_emb():
            def rotary_emb(positions, query, key, head_size,
                           cos_sin_cache, is_neox_style):
                # Assume is_neox_style == True, disable assert for performance
                # assert is_neox_style
                cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
                cos_repeated = cos.contiguous().repeat(1, 1, 2).unsqueeze(-2)
                sin_repeated = sin.contiguous().repeat(1, 1, 2).unsqueeze(-2)
                query_reshaped = query.view(*cos_repeated.shape[:2], query.shape[-1] // head_size, head_size).clone()
                key_reshaped = key.view(*cos_repeated.shape[:2], key.shape[-1] // head_size, head_size).clone()
                # bug, aclnnApplyRotaryPosEmb doesn't support non-contiguous query and key
                # which CANN document say it can support
                ext.rotary_embedding_v2(query_reshaped, key_reshaped, cos_repeated, sin_repeated, head_size)
                query.copy_(query_reshaped.view(*query.shape))
                key.copy_(key_reshaped.view(*key.shape))
                return

            vllm_ops.rotary_embedding = rotary_emb

        def patch_silu_and_mul():
            def silu_and_mul(out, x):
                # no fusion implement
                d = x.size(-1) // 2
                torch.mul(torch.nn.functional.silu(x[..., :d]), x[..., d:], out=out)
                return

            vllm_ops.silu_and_mul = silu_and_mul

        def patch_reshape_and_cache():
            def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, kv_scale):
                ext.dest_index_copy_kv(key.view(key.size(0), -1), slot_mapping, key_cache.view(-1, key_cache.size(-1)))
                ext.dest_index_copy_kv(value.view(value.size(0), -1), slot_mapping, value_cache.view(-1, value_cache.size(-1)))
                return

            vllm_cache_ops.reshape_and_cache = reshape_and_cache

        def patch_run_memory_efficient_xformers_forward():
            @lru_cache(None)
            def cached_mask_generator(q_seq_len, kv_seq_len):
                mask = torch.tril(torch.ones(q_seq_len, kv_seq_len,
                                             dtype=torch.bool), diagonal=0).cuda()
                return torch.logical_not(mask)

            def batch_same_seq_len_prefill(out, query, key, value, dim, batch,
                                           num_q_head, num_kv_head, prefill_meta):
                seq_len = prefill_meta.max_prompt_len
                mask = cached_mask_generator(seq_len, seq_len)
                ext.prompt_flash_attention(out.view(batch, -1, dim * num_q_head),
                                           query.view(batch, -1, dim * num_q_head),
                                           key.view(batch, -1, dim * num_kv_head),
                                           value.view(batch, -1, dim * num_kv_head),
                                           mask, [], seq_len, num_q_head, num_kv_head, dim)
                return

            def batch_different_seq_len_prefill(out, query, key, value, dim, batch,
                                                num_q_head, num_kv_head, prefill_meta):
                for i in range(batch):
                    start = prefill_meta.seq_start_loc[i]
                    end = start + prefill_meta.prompt_lens_tensor[i]
                    single_seq_len = prefill_meta.prompt_lens[i]
                    single_q = query[start:end].view(1, single_seq_len, -1).clone()
                    single_k = key[start:end].view(1, single_seq_len, -1).clone()
                    single_v = value[start:end].view(1, single_seq_len, -1).clone()
                    single_out = out[start:end, :].view(1, single_seq_len, -1)
                    mask = cached_mask_generator(single_seq_len, single_seq_len)
                    ext.prompt_flash_attention(single_out, single_q, single_k, single_v, mask, [],
                                               single_seq_len, num_q_head, num_kv_head, dim)
                return

            def run_memory_efficient_xformers_forward(self, query, key, value, prefill_meta):
                out = torch.empty_like(query)
                num_q_head, num_kv_head = query.size(1), key.size(1)
                dim, batch = query.size(2), len(prefill_meta.prompt_lens)
                if prefill_meta.max_prompt_len * len(prefill_meta.prompt_lens) == query.size(0):
                    batch_same_seq_len_prefill(out, query, key, value, dim, batch,
                                               num_q_head, num_kv_head, prefill_meta)
                else:
                    batch_different_seq_len_prefill(out, query, key, value, dim, batch,
                                                    num_q_head, num_kv_head, prefill_meta)
                return out

            vllm_attn_xformers.XFormersImpl._run_memory_efficient_xformers_forward \
                = run_memory_efficient_xformers_forward

        def patch_page_attn():
            @staticmethod
            def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
                return (2, num_blocks, block_size, num_kv_heads * head_size)

            @staticmethod
            def split_kv_cache(kv_cache, num_kv_heads, head_size):
                return kv_cache[0], kv_cache[1]
            
            @staticmethod
            def forward_decode(query, key_cache, value_cache, block_tables, context_lens, max_context_len,
                               kv_cache_dtype, num_kv_heads, scale, alibi_slopes, kv_scale):
                out = torch.ones_like(query)
                actual_seq_len = context_lens.tolist()
                q_shape = (query.size(0), -1)
                kv_shape = (-1, key_cache.size(-1))
                ext.paged_attention(out.view(*q_shape), query.view(*q_shape), key_cache.view(*kv_shape),
                                    value_cache.view(*kv_shape), None, actual_seq_len,
                                    query.size(-2), num_kv_heads, query.size(-1),
                                    block_tables, key_cache.size(1))
                return out

            vllm_paged_attn.PagedAttention.get_kv_cache_shape = get_kv_cache_shape
            vllm_paged_attn.PagedAttention.split_kv_cache = split_kv_cache
            vllm_paged_attn.PagedAttention.forward_decode = forward_decode


        try:
            locals()[f"patch_{op}"]()
            print(f"[deeplink_ext] patched diopi implementation of {op}\n", end="")
        except KeyError:
            print(
                f"[deeplink_ext] unknow op: {op}, supported ops: {DEFAULT_PATCH_LIST}\n",
                end="",
            )
        except AttributeError:
            print(f"[deeplink_ext] op {op} is not implemented in diopi\n", end="")

    for op in patch_list:
        try_patch(op)

_patch_vllm()
