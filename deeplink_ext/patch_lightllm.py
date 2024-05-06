# Copyright (c) 2024, DeepLink.

import torch
import torch_dipu


def _patch_lightllm():
    import os
    import deeplink_ext.cpp_extensions as ext
    import lightllm.common.basemodel.triton_kernel.destindex_copy_kv as destindex_copy_kv_pack  # type: ignore
    # import lightllm.common.basemodel.triton_kernel.apply_penalty as apply_penalty_pack  # type: ignore
    import lightllm.models.llama.triton_kernel.context_flashattention_nopad as context_attention_pack  # type: ignore
    import lightllm.models.llama.triton_kernel.token_attention_nopad_att1 as token_attention_pack  # type: ignore
    import lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev as token_attention_softmax_reducev_pack  # type: ignore
    import lightllm.models.llama.triton_kernel.rmsnorm as rms_norm_pack  # type: ignore
    import lightllm.models.llama.triton_kernel.rotary_emb as rotary_emb_pack  # type: ignore
    import math

    DEFAULT_PATCH_LIST = [
        "dest_index_copy_kv",
        # "apply_penalty",
        "context_attention_inference",
        "token_attention_inference",
        "token_softmax_reducev_inference",
        "rms_norm_lightllm",
        "rotary_emb",
    ]
    mask_cache = {}
    PATCH_LIST_ENV_NAME = "DEEPLINKEXT_LIGHTLLM_PATCH_LIST"
    patch_list_env = os.environ.get(PATCH_LIST_ENV_NAME)
    use_custom_patch_list = patch_list_env is not None
    patch_list = (
        patch_list_env.split(",") if use_custom_patch_list else DEFAULT_PATCH_LIST
    )
    if use_custom_patch_list:
        print(f"[deeplink_ext] use custom lightllm patch list: {patch_list}\n", end="")

    def try_patch(op: str):
        def patch_dest_index_copy_kv():
            destindex_copy_kv_pack.destindex_copy_kv = ext.dest_index_copy_kv

        # def patch_apply_penalty():
        #     apply_penalty_pack.apply_penalty = ext.apply_penalty

        def patch_context_attention_inference():
            from torch.profiler import record_function

            @record_function('loop_context_attention')
            def flash_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
                batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]

                scale = 1 / math.sqrt(dim)
                for i in range(batch):
                    start = b_start_loc[i]
                    end = start + b_seq_len[i]

                    single_seq_len = int(b_seq_len[i])
                    single_q = q[start:end].view(1, single_seq_len, -1)
                    single_k = k[start:end].view(1, single_seq_len, -1)
                    single_v = v[start:end].view(1, single_seq_len, -1)

                    single_out = out[start:end, :].view(1, single_seq_len, -1)
                    key_str = f"bs:1-seqlen:{single_seq_len}"
                    if key_str not in mask_cache:
                        mask = torch.tril(torch.ones(single_seq_len, single_seq_len, dtype=torch.bool), diagonal=0).cuda()
                        mask = mask.repeat(1, 1, 1)
                        mask = torch.logical_not(mask)
                        mask_cache[key_str] = mask
                        print(f"{key_str}: cache mask in context attention")
                    mask = mask_cache[key_str]
                    with torch.profiler.record_function("PromptFA-B1"):
                        ext.prompt_flash_attention(single_out, single_q, single_k, single_v, None, mask, [], head, scale, 2147473647, 0, "BSH", 0)
                return out

            @record_function('fused_context_attention')
            def fused_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
                batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]
                scale = 1 / math.sqrt(dim)
                total_q = torch.empty([batch, max_input_len, head * dim], dtype=torch.float16, device='cuda')
                total_k = torch.empty_like(total_q)
                total_v = torch.empty_like(total_q)
                total_out = torch.empty_like(total_q)

                for i in range(batch):
                    start = b_start_loc[i]
                    end = start + b_seq_len[i]

                    assert max_input_len == int(b_seq_len[i])
                    total_q[i] = q[start:end].view(max_input_len, -1)
                    total_k[i] = k[start:end].view(max_input_len, -1)
                    total_v[i] = v[start:end].view(max_input_len, -1)

                key_str = f"bs:{batch}-seqlen:{max_input_len}"
                if key_str not in mask_cache:
                    mask = torch.tril(torch.ones(max_input_len, max_input_len, dtype=torch.bool), diagonal=0).cuda()
                    mask = mask.repeat(batch, 1, 1)
                    mask = torch.logical_not(mask)
                    mask_cache[key_str] = mask
                    print(f"{key_str}: cache mask in context attention")
                mask = mask_cache[key_str]
                with torch.profiler.record_function(f"PromptFA-B{batch}"):
                    ext.prompt_flash_attention(total_out, total_q, total_k, total_v, None, mask, [], head, scale, 2147473647, 0, "BSH", 0)
                for i in range(batch):
                    start = b_start_loc[i]
                    end = start + b_seq_len[i]
                    out[start:end, :] = total_out[i].reshape(-1, head, dim)
                torch.cuda.synchronize()
                return out

            def test_fa(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
                out2 = torch.empty_like(out)

                flash_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len)
                fused_context_attention(q, k, v, out2, b_start_loc, b_seq_len, max_input_len)
                assert torch.allclose(out.cpu(), out2.cpu(), rtol=1e-2, atol=1e-2)

                flash_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len)
                fused_context_attention(q, k, v, out2, b_start_loc, b_seq_len, max_input_len)
                assert torch.allclose(out.cpu(), out2.cpu(), rtol=1e-2, atol=1e-2)

                torch.cuda.synchronize()
                path = "./tmo_profile"
                with torch_dipu.profiler.NativeProfile(path, False):
                    flash_context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len)
                    fused_context_attention(q, k, v, out2, b_start_loc, b_seq_len, max_input_len)
                    torch.cuda.synchronize()

                print("context attention success.")
                quit()
                return out

            context_attention_pack.context_attention_fwd = (
                # ext.context_attention_inference
                flash_context_attention
            )

        def patch_token_attention_inference():
            token_attention_pack.token_att_fwd = ext.token_attention_inference
            token_attention_pack.token_decode_attention_fwd = ext.token_decode_attention_inference
            # token_attention_pack.token_decode_attention_fwd = ext.token_decode_attention_inference_batch_one#ext.token_decode_attention_inference

        def patch_token_softmax_reducev_inference():
            token_attention_softmax_reducev_pack.token_softmax_reducev_fwd = (
                ext.token_softmax_reducev_inference
            )

        def patch_rms_norm_lightllm():
            def rms_norm(input, weight, eps):
                output = torch.empty_like(input)
                inv_rms_shape = list(input.shape[:-1]) + [1]
                inv_rms = torch.empty(
                    inv_rms_shape, dtype=torch.float32, device=input.device
                )
                ext.rms_norm(output, inv_rms, input, None, weight, None, eps)

                return output

            rms_norm_pack.rmsnorm_forward = rms_norm

        def patch_rotary_emb():
            def rotary_emb(q, cos, sin):
                seq_len = q.shape[0]
                dim = q.shape[-1]
                cos_view = cos.view([seq_len, 1, dim // 2])
                sin_view = sin.view([seq_len, 1, dim // 2])
                ext.apply_rotary(q, q, cos_view, sin_view, False, False)

            rotary_emb_pack.rotary_emb_fwd = rotary_emb

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


_patch_lightllm()
