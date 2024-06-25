# Copyright (c) 2024, DeepLink.

import torch
import torch_dipu


def _patch_lmdeploy():
    import os
    import sys
    from argparse import Namespace
    from functools import lru_cache
    import deeplink_ext.cpp_extensions as ext
    import lmdeploy.pytorch.engine.devices as engine_devices_pack

    DEFAULT_PATCH_LIST = [
        "kernels",
        "engine_devices",
    ]
    PATCH_LIST_ENV_NAME = "DEEPLINKEXT_LMDEPLOY_PATCH_LIST"
    patch_list_env = os.environ.get(PATCH_LIST_ENV_NAME)
    use_custom_patch_list = patch_list_env is not None
    patch_list = (
        patch_list_env.split(",") if use_custom_patch_list else DEFAULT_PATCH_LIST
    )
    if use_custom_patch_list:
        print(f"[deeplink_ext] use custom lmdeploy patch list: {patch_list}\n", end="")

    def try_patch(op: str):
        def patch_kernels():
            def rms_norm(
                hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
            ):
                output = torch.empty_like(hidden_states)
                inv_rms_shape = list(hidden_states.shape[:-1]) + [1]
                inv_rms = torch.empty(
                    inv_rms_shape, dtype=torch.float32, device=hidden_states.device
                )
                ext.rms_norm(
                    output, inv_rms, hidden_states, weight.shape, weight, None, eps
                )
                return output

            def apply_rotary_pos_emb(
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                position_ids: torch.Tensor,
                position_ids_1d: torch.Tensor,
                q_embed=None,
                k_embed=None,
            ):
                bs, head, dim = query_states.shape
                numKeyValueHeads = key_states.shape[1]
                seqlens = [
                    (min(position_id), max(position_id) + 1)
                    for position_id in position_ids.tolist()
                ]
                query_states = query_states.reshape(bs, head * dim)
                key_states = key_states.reshape(bs, numKeyValueHeads * dim)
                cos = torch.cat([cos[i:j] for i, j in seqlens]).view(1, bs, 1, -1)
                sin = torch.cat([sin[i:j] for i, j in seqlens]).view(1, bs, 1, -1)
                ext.rotary_embedding_v2(query_states, key_states, cos, sin, dim)
                return query_states.view(bs, head, dim), key_states.view(
                    bs, numKeyValueHeads, dim
                )

            def fused_rotary_emb(
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                position_ids: torch.LongTensor,
                inv_freq: torch.Tensor,
                scaling_factor: float,
                out_q: torch.Tensor = None,
                out_k: torch.Tensor = None,
            ):
                _, bs, head, dim = query_states.shape
                _, _, numKeyValueHeads, _ = key_states.shape
                query_states = query_states.view(bs, head * dim)
                key_states = key_states.view(bs, numKeyValueHeads * dim)
                position_ids = position_ids.squeeze(0).unsqueeze(-1)
                pos_freq = position_ids / scaling_factor * inv_freq
                cos = (
                    torch.cos(pos_freq)
                    .view(position_ids.shape[0], 1, -1)
                    .repeat(1, 1, 2)
                    .to(query_states.dtype)
                )
                sin = (
                    torch.sin(pos_freq)
                    .view(position_ids.shape[0], 1, -1)
                    .repeat(1, 1, 2)
                    .to(query_states.dtype)
                )
                ext.rotary_embedding_v2(query_states, key_states, cos, sin, dim)
                query_states = query_states.view(1, bs, head, dim)
                key_states = key_states.view(1, bs, numKeyValueHeads, dim)
                return query_states, key_states

            def fill_kv_cache(
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                key_caches: torch.Tensor,
                value_caches: torch.Tensor,
                q_start_loc: torch.Tensor,
                q_seq_length: torch.Tensor,
                kv_seq_length: torch.Tensor,
                max_q_seq_length: int,
                block_offsets: torch.Tensor,
            ):
                """fill key/value state to cache for paged attention."""
                assert hasattr(
                    engine_devices_pack.ascend.ASCENDDeviceUtils.step_context,
                    "kv_start_indices",
                )
                kv_start_indices = (
                    engine_devices_pack.ascend.ASCENDDeviceUtils.step_context.kv_start_indices
                )
                engine_devices_pack.ascend.ASCENDDeviceUtils.add_kv_states(
                    key_states, value_states
                )
                dest_index_copy_kv(key_states, kv_start_indices, key_caches)
                dest_index_copy_kv(value_states, kv_start_indices, value_caches)

            def dest_index_copy_kv(states, dest_loc, caches):
                block_num, block_size, head, dim = caches.size()
                caches_tmp = caches.view(block_num * block_size, head, dim)
                ext.dest_index_copy_kv(states, dest_loc, caches_tmp)
                caches[:] = caches_tmp.view(block_num, block_size, head, dim)

            @lru_cache(None)
            def cached_mask_generator(q_seq_len, kv_seq_len):
                mask = torch.tril(
                    torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool),
                    diagonal=kv_seq_len - q_seq_len,
                ).cuda()
                return torch.logical_not(mask)

            def paged_attention_fwd(
                query_states: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                attn_output: torch.Tensor,
                block_offsets: torch.Tensor,
                q_start_loc: torch.Tensor,
                q_seqlens: torch.Tensor,
                kv_seqlens: torch.Tensor,
                max_seqlen: int,
            ):
                is_decoding = query_states.shape[-3] == q_seqlens.size(0)
                block_num, block_size, head, dim = key_cache.size()
                kv_cache_len = block_num * block_size
                k = key_cache.reshape(block_num * block_size, head, dim)
                v = value_cache.reshape(block_num * block_size, head, dim)
                if not is_decoding:
                    key_states = engine_devices_pack.ascend.ASCENDDeviceUtils.key_states
                    value_states = (
                        engine_devices_pack.ascend.ASCENDDeviceUtils.value_states
                    )
                    flash_context_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_output,
                        k,
                        v,
                        block_offsets.to(torch.int32),
                        q_start_loc,
                        q_seqlens.tolist(),
                        kv_seqlens.tolist(),
                        block_size,
                        kv_cache_len,
                    )
                else:
                    paged_token_attention(
                        query_states,
                        k,
                        v,
                        attn_output,
                        kv_seqlens.tolist(),
                        block_offsets.to(torch.int32),
                        block_size,
                    )

            def flash_context_attention(
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                attn_output: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                block_offsets: torch.Tensor,
                q_start_loc: torch.Tensor,
                q_seqlens: list,
                kv_seqlens: list,
                block_size: int,
                kv_cache_len: int,
            ):
                batch, head, dim = (
                    q_start_loc.shape[0],
                    query_states.shape[1],
                    query_states.shape[2],
                )
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
                        mask = cached_mask_generator(single_seqlen, single_seqlen)
                        ext.prompt_flash_attention(
                            single_out,
                            single_q,
                            single_k,
                            single_v,
                            mask,
                            [kv_seqlens[i]],
                            kv_seqlens[i],
                            head,
                            numKeyValueHeads,
                            dim,
                        )
                    else:
                        key_cache = key_cache.reshape(
                            1, kv_cache_len, numKeyValueHeads * dim
                        )
                        value_cache = value_cache.reshape(
                            1, kv_cache_len, numKeyValueHeads * dim
                        )
                        for j in range(q_seqlens[i]):
                            single_q = query_states[start + j : start + j + 1].view(
                                1, 1, -1
                            )
                            single_out = attn_output[start + j : start + j + 1].view(
                                1, 1, -1
                            )
                            mask = cached_mask_generator(q_seqlens[i], kv_seqlens[i])
                            ext.paged_attention(
                                single_out,
                                single_q,
                                key_cache,
                                value_cache,
                                mask[j : j + 1],
                                [kv_seqlens[i]],
                                head,
                                numKeyValueHeads,
                                dim,
                                block_offsets[i : i + 1],
                                block_size,
                            )

            def paged_token_attention(
                q,
                k_cache,
                v_cache,
                attn_output,
                kv_seqlens,
                block_table: torch.Tensor,
                block_size,
            ):
                numKeyValueHeads = k_cache.shape[1]
                assert k_cache.shape[1] == v_cache.shape[1]
                bs, head, dim = q.shape
                kv_cache_len = k_cache.shape[0]
                q = q.reshape(bs, 1, head * dim)
                k_cache = k_cache.reshape(1, kv_cache_len, numKeyValueHeads * dim)
                v_cache = v_cache.reshape(1, kv_cache_len, numKeyValueHeads * dim)
                ext.paged_attention(
                    attn_output.view(q.shape),
                    q,
                    k_cache,
                    v_cache,
                    None,
                    kv_seqlens,
                    head,
                    numKeyValueHeads,
                    dim,
                    block_table,
                    block_size,
                )

            def multinomial_sampling(
                scores: torch.Tensor,
                seeds: torch.LongTensor,
                offsets: torch.LongTensor,
                indices: torch.Tensor = None,
            ):
                sampled_index = torch.multinomial(
                    scores, num_samples=1, replacement=True
                )
                outputs = torch.gather(indices, dim=1, index=sampled_index)
                return outputs.view(-1)

            ascend_kernels = Namespace(
                **{
                    "rms_norm": rms_norm,
                    "apply_rotary_pos_emb": apply_rotary_pos_emb,
                    "fused_rotary_emb": fused_rotary_emb,
                    "fill_kv_cache": fill_kv_cache,
                    "paged_attention_fwd": paged_attention_fwd,
                    "multinomial_sampling": multinomial_sampling,
                }
            )
            sys.modules["lmdeploy.pytorch.kernels.ascend"] = ascend_kernels

        def patch_engine_devices():
            class ASCENDDeviceUtils(engine_devices_pack.BaseDeviceUtils):

                device = "ascend"

                @classmethod
                def update_step_context(cls, step_context):
                    """update step context."""
                    kv_start_indices = []
                    _, block_size, _, _ = step_context.kv_caches[0][0].shape
                    for i in range(step_context.q_start_loc.size(0)):
                        history_length = step_context.history_lengths[i]
                        block_idx = history_length // block_size
                        block_loc = step_context.block_offsets[i][block_idx]
                        token_loc = history_length % block_size
                        for _ in range(step_context.q_seq_length[i]):
                            kv_start_indices.append(block_loc * block_size + token_loc)
                            if _ == step_context.q_seq_length[i] - 1:
                                break
                            token_loc = (token_loc + 1) % block_size
                            block_idx = block_idx if token_loc else block_idx + 1
                            block_loc = step_context.block_offsets[i][block_idx]
                    step_context.kv_start_indices = torch.tensor(
                        kv_start_indices, device=step_context.block_offsets.device
                    )
                    cls.step_context = step_context
                    return step_context

                @classmethod
                def add_kv_states(cls, key_states, value_states):
                    cls.key_states = key_states
                    cls.value_states = value_states

            engine_devices = Namespace(**{"ASCENDDeviceUtils": ASCENDDeviceUtils})
            setattr(engine_devices_pack, "ascend", engine_devices)
            sys.modules["lmdeploy.pytorch.engine.devices.ascend"] = engine_devices

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


_patch_lmdeploy()
