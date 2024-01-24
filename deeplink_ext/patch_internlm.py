# Copyright (c) 2024, DeepLink.


def _patch_internlm():
    import importlib.util
    import os
    import sys
    import unittest.mock as mock
    import deeplink_ext.internlm_ops as ext

    def _find_or_mock_module(module_name):
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            sys.modules[module_name] = mock.Mock()

    def _find_flash_attn():
        flash_attn_spec = importlib.util.find_spec("flash_attn")
        if flash_attn_spec is None:
            internlm_spec = importlib.util.find_spec("internlm")
            assert internlm_spec is not None
            assert internlm_spec.submodule_search_locations is not None
            sys.path.append(
                os.path.abspath(
                    os.path.join(
                        internlm_spec.submodule_search_locations[0],
                        "../third_party/flash-attention",
                    )
                )
            )

    def _patch_flash_attn():
        import flash_attn.losses.cross_entropy
        import torch.nn

        def CrossEntropyLossProxy(reduction, **_):
            return torch.nn.CrossEntropyLoss(reduction=reduction)

        flash_attn.losses.cross_entropy.CrossEntropyLoss = CrossEntropyLossProxy

        import flash_attn.modules.mha

        flash_attn.modules.mha.SelfAttention = ext.mha.DeepLinkSelfAttention
        flash_attn.modules.mha.FlashSelfAttention = ext.mha.DeepLinkSelfAttention
        flash_attn.modules.mha.CrossAttention = ext.mha.DeepLinkCrossAttention
        flash_attn.modules.mha.FlashCrossAttention = ext.mha.DeepLinkCrossAttention

    def _patch_ops():
        import internlm.model.embedding

        internlm.model.embedding.apply_rotary_emb_qkv_ = (
            ext.rotary.DeepLinkApplyRotaryEmbQKV_.apply
        )
        internlm.model.embedding.legacy_apply_rotary_embed = (
            ext.rotary.DeepLinkApplyRotaryEmb.apply
        )
        internlm.model.embedding.legacy_apply_rotary_embed_qkv = (
            ext.rotary.DeepLinkApplyRotaryEmbQKV_.apply
        )

        import internlm.model.norm

        internlm.model.norm.RMSNormTorch = (
            ext.rms_norm.DeepLinkRMSNormWithNormalizedShape
        )

    _find_or_mock_module("rotary_emb")
    _find_or_mock_module("fused_dense_lib")
    _find_or_mock_module("xentropy_cuda_lib")
    _find_or_mock_module("flash_attn_cuda")
    _find_flash_attn()
    _patch_flash_attn()
    _patch_ops()
    print("[deeplink_ext] patched diopi implementation of internlm")


_patch_internlm()
