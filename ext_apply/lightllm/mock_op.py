import torch
import os
try:
    import dipu_ext.ext_ as ext
    default_diopi_mock_op_list = ['dest_index_copy_kv', 'token_attention_inference',
                                  'token_softmax_reducev_inference',
                                  'context_attention_inference', 'apply_penalty']
    diopi_mock_op_list = os.environ.get('diopi_mock_op_list').split(
        ',') if 'diopi_mock_op_list' in os.environ else default_diopi_mock_op_list
    print(f"diopi_mock_op_list:{diopi_mock_op_list}")
    if hasattr(ext, 'dest_index_copy_kv') and 'dest_index_copy_kv' in diopi_mock_op_list:
        import lightllm.common.basemodel.triton_kernel.destindex_copy_kv as destindex_copy_kv_pack
        destindex_copy_kv_pack.destindex_copy_kv = ext.dest_index_copy_kv
        print("use diopi_dest_index_copy_kv as destindex_copy_kv")

    if hasattr(ext, 'apply_penalty') and 'apply_penalty' in diopi_mock_op_list:
        import lightllm.common.basemodel.triton_kernel.apply_penalty as apply_penalty_pack
        apply_penalty_pack.apply_penalty = ext.apply_penalty
        print("use diopi_apply_penalty as apply_penalty")

    if hasattr(ext, 'context_attention_inference') and 'context_attention_inference' in diopi_mock_op_list:
        import lightllm.models.llama.triton_kernel.context_flashattention_nopad as context_attention_pack
        context_attention_pack.context_attention_fwd = ext.context_attention_inference
        print("use diopi_context_attention_inference as context_attention_fwd")

    if hasattr(ext, 'token_attention_inference') and 'token_attention_inference' in diopi_mock_op_list:
        import lightllm.models.llama.triton_kernel.token_attention_nopad_att1 as token_attention_pack
        token_attention_pack.token_att_fwd = ext.token_attention_inference
        print("use diopi_token_attention_inference as token_att_fwd")

    if hasattr(ext, 'token_softmax_reducev_inference') and 'token_softmax_reducev_inference' in diopi_mock_op_list:
        import lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev as token_attention_softmax_reducev_pack
        token_attention_softmax_reducev_pack.token_softmax_reducev_fwd = ext.token_softmax_reducev_inference
        print("use diopi_token_softmax_reducev_inference as token_softmax_reducev_fwd")

    if hasattr(ext, 'rms_norm_lightllm') and 'rms_norm_lightllm' in diopi_mock_op_list:
        import lightllm.models.llama.triton_kernel.rmsnorm as rms_norm_pack
        rms_norm_pack.rmsnorm_forward = ext.rms_norm_lightllm
        print("use diopi_rms_norm as rmsnorm_forward")

    if hasattr(ext, 'rotary_emb') and 'rotary_emb' in diopi_mock_op_list:
        import lightllm.models.llama.triton_kernel.rotary_emb as rotary_emb_pack
        rotary_emb_pack.rotary_emb_fwd = ext.rotary_emb
        print("use diopi_rotary_embedding as rotary_emb_fwd")
except ImportError:
    pass
