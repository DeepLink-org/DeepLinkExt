import torch
import os
try:
    #export DIOPI_OP_LIST=dest_index_copy_kv,token_attention_inference,token_softmax_reducev_inference,context_attention_inference,apply_penalty
    import dipu_ext.ext_ as ext
    diopi_op_list = os.environ['DIOPI_OP_LIST'].split(',')
    print(f"DIOPI_OP_LIST:{diopi_op_list}")
    if hasattr(ext, 'dest_index_copy_kv') and 'dest_index_copy_kv' in diopi_op_list:
        import lightllm.common.basemodel.triton_kernel.destindex_copy_kv as destindex_copy_kv_pack
        destindex_copy_kv_pack.destindex_copy_kv = ext.dest_index_copy_kv
        print("use diopi dest_index_copy_kv as destindex_copy_kv")

    if hasattr(ext, 'apply_penalty') and 'apply_penalty' in diopi_op_list:
        import lightllm.common.basemodel.triton_kernel.apply_penalty as apply_penalty_pack
        apply_penalty_pack.apply_penalty = ext.apply_penalty
        print("use diopi apply_penalty")

    if hasattr(ext, 'context_attention_inference') and 'context_attention_inference'  in diopi_op_list:
        import lightllm.models.llama.triton_kernel.context_flashattention_nopad as context_attention_pack
        context_attention_pack.context_attention_fwd = ext.context_attention_inference
        print("use diopi context_attention_inference as context_attention_fwd")
    if hasattr(ext, 'token_attention_inference') and 'token_attention_inference' in diopi_op_list:
        import lightllm.models.llama.triton_kernel.token_attention_nopad_att1 as token_attention_pack
        token_attention_pack.token_att_fwd = ext.token_attention_inference
        print("use diopi token_attention_inference as token_att_fwd")
    if hasattr(ext, 'token_softmax_reducev_inference') and 'token_softmax_reducev_inference' in diopi_op_list:
        import lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev as token_attention_softmax_reducev_pack
        token_attention_softmax_reducev_pack.token_softmax_reducev_fwd = ext.token_softmax_reducev_inference
        print("use diopi token_softmax_reducev_inference as token_softmax_reducev_fwd")
except ImportError:
    pass


