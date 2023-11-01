import torch

try:
    import dipu_ext.ext_ as ext

    if hasattr(ext, 'dest_index_copy_kv'):
        import lightllm.common.basemodel.triton_kernel.destindex_copy_kv as destindex_copy_kv_pack
        destindex_copy_kv_pack.destindex_copy_kv = ext.dest_index_copy_kv
        print("use diopi dest_index_copy_kv")
except ImportError:
    pass


#ext.context_attention_inference(      ext.dest_index_copy_kv(               ext.token_attention_inference(        ext.token_softmax_reducev_inference