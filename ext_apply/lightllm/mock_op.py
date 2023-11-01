import torch

try:
    import dipu_ext.ext_ as ext

    """
    if hasattr(ext, 'dest_index_copy_kv'):
        import lightllm.common.basemodel.triton_kernel.destindex_copy_kv as destindex_copy_kv_pack
        destindex_copy_kv_pack.destindex_copy_kv = ext.dest_index_copy_kv
        print("use diopi dest_index_copy_kv as destindex_copy_kv")

    if hasattr(ext, 'apply_penalty'):
        import lightllm.common.basemodel.triton_kernel.apply_penalty as apply_penalty_pack
        apply_penalty_pack.apply_penalty = ext.apply_penalty
        print("use diopi apply_penalty")

    """
    if hasattr(ext, 'context_attention_inference'):
        import lightllm.models.llama.triton_kernel.context_flashattention_nopad as context_attention_pack
        context_attention_pack.context_attention_fwd = ext.context_attention_inference
        print("use diopi context_attention_inference as context_attention_fwd")
    """
    if hasattr(ext, 'token_attention_inference'):
        import lightllm.models.llama.triton_kernel.token_attention_nopad_att1 as token_attention_pack
        token_attention_pack.token_att_fwd = ext.token_attention_inference
        print("use diopi token_attention_inference as token_att_fwd")
    if hasattr(ext, 'token_softmax_reducev_inference'):
        import lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev as token_attention_softmax_reducev_pack
        token_attention_softmax_reducev_pack.token_softmax_reducev_fwd = ext.token_softmax_reducev_inference
        print("use diopi token_softmax_reducev_inference as token_softmax_reducev_fwd")
    """
except ImportError:
    pass


#ext.context_attention_inference(   apply_penalty   ext.dest_index_copy_kv(               ext.token_attention_inference(        ext.token_softmax_reducev_inference