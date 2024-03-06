# Copyright (c) 2024, DeepLink.

import os

_force_fallback = os.environ.get("DEEPLINK_EXT_FORCE_FALLBACK", "0") != "0"


def _patch_internlm(force_fallback: bool = False):
    import importlib.util
    import sys
    import types
    import torch

    def _find_or_mock_module(module_name) -> bool:
        """Find or mock a module. Return True if the module is found, False otherwise."""
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            sys.modules[module_name] = types.SimpleNamespace()  # type: ignore
        return module_spec is not None

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

    def _force_fallback():
        print(
            "[deeplink_ext] force_fallback is set, removing everything from cpp_extensions"
        )
        import deeplink_ext.cpp_extensions as cpp_ext

        for attr in dir(cpp_ext):
            if not attr.startswith("__") and callable(getattr(cpp_ext, attr)):
                print(f"[deeplink_ext] removing {attr} from cpp_extensions")
                delattr(cpp_ext, attr)

    def _patch_flash_attn():
        import deeplink_ext.internlm_ops as ext
        import flash_attn.losses.cross_entropy  # type: ignore
        import torch.nn

        def CrossEntropyLossProxy(reduction, **_):
            return torch.nn.CrossEntropyLoss(reduction=reduction)

        flash_attn.losses.cross_entropy.CrossEntropyLoss = CrossEntropyLossProxy

        import flash_attn.modules.mha  # type: ignore

        flash_attn.modules.mha.SelfAttention = ext.mha.DeepLinkSelfAttention
        flash_attn.modules.mha.FlashSelfAttention = ext.mha.DeepLinkSelfAttention
        flash_attn.modules.mha.CrossAttention = ext.mha.DeepLinkCrossAttention
        flash_attn.modules.mha.FlashCrossAttention = ext.mha.DeepLinkCrossAttention

    def _patch_ops():
        import deeplink_ext.internlm_ops as ext
        import internlm.model.embedding  # type: ignore

        class NonLegacyRotaryEmbQKV_(torch.autograd.Function):
            """the first 2 dims of qkv has been squeezed"""

            @staticmethod
            def forward(ctx, qkv: torch.Tensor, *args, **kwargs):  # type: ignore
                unsqueezed_qkv = qkv.view([1] + list(qkv.shape))
                out: torch.Tensor = ext.rotary.DeepLinkApplyRotaryEmbQKV_.forward(
                    ctx, unsqueezed_qkv, *args, **kwargs
                )
                return out.view(out.shape[1:])

            @staticmethod
            def backward(ctx, dqkv: torch.Tensor, *args, **kwargs):  # type: ignore
                unqueezed_dqkv = dqkv.view([1] + list(dqkv.shape))
                out: tuple = ext.rotary.DeepLinkApplyRotaryEmbQKV_.backward(
                    ctx, unqueezed_dqkv, *args, **kwargs
                )
                return (out[0].view(out[0].shape[1:]),) + out[1:]

        internlm.model.embedding.apply_rotary_emb_qkv_ = NonLegacyRotaryEmbQKV_.apply
        internlm.model.embedding.legacy_apply_rotary_embed = (
            ext.rotary.DeepLinkApplyRotaryEmb.apply
        )
        internlm.model.embedding.legacy_apply_rotary_embed_qkv = (
            ext.rotary.DeepLinkApplyRotaryEmbQKV_.apply
        )

        import internlm.model.norm  # type: ignore

        # NOTE: RMSNormTorch class object has been assigned to RMSNorm via
        #           RMSNorm = try_import_RMSNorm()
        #       everywhere (e.g. in modeling_llama.py).
        #       Thus simply reassigning RMSNormTorch to DeepLinkRMSNorm won't work.
        #       And we don't want to reassign every RMSNorm to DeepLinkRMSNorm.
        #       So we patch RMSNormTorch.__new__ to create a DeepLinkRMSNorm instance
        #       whenever RMSNorm(...) is called.
        internlm.model.norm.RMSNormTorch.__new__ = lambda _, *args, **kwargs: (
            ext.rms_norm.DeepLinkRMSNormWithNormalizedShape(*args, **kwargs)
        )

    cpp_ext_found = _find_or_mock_module("deeplink_ext.cpp_extensions")
    if not cpp_ext_found:
        print(
            "[deeplink_ext] WARNING: cpp_extensions not compiled, falling back to pure python implementation"
        )
    _find_or_mock_module("rotary_emb")
    _find_or_mock_module("fused_dense_lib")
    _find_or_mock_module("xentropy_cuda_lib")
    _find_or_mock_module("flash_attn_cuda")
    _find_flash_attn()
    if force_fallback:
        _force_fallback()
    _patch_flash_attn()
    _patch_ops()
    print("[deeplink_ext] patched diopi implementation of internlm\n", end="")


_patch_internlm(force_fallback=_force_fallback)

__all__ = []
