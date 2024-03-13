# Copyright (c) 2024, DeepLink.

import os

__all__ = []

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
        try:
            import deeplink_ext.cpp_extensions as cpp_ext
        except Exception as e:
            print(
                "[deeplink_ext] WARNING: failed to import deeplink_ext.cpp_extensions, "
                "so everything will be falled back to pure python implementation. "
                "Please check this import failure if you are using torch_dipu."
            )
            return

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
        import flash_attn.layers.rotary  # type: ignore
        import internlm.model.embedding  # type: ignore

        flash_attn.layers.rotary.apply_rotary = ext.rotary.apply_rotary

        class NonLegacyRotaryEmbQKV_(torch.autograd.Function):
            """the first 2 dims of qkv has been squeezed"""

            @staticmethod
            def forward(ctx, qkv: torch.Tensor, *args, **kwargs):  # type: ignore
                unsqueezed_qkv = qkv.view([1] + list(qkv.shape))
                out: torch.Tensor = (
                    internlm.model.embedding.LegacyApplyRotaryEmbQKV_.forward(
                        ctx, unsqueezed_qkv, *args, **kwargs
                    )
                )
                return out.view(out.shape[1:])

            @staticmethod
            def backward(ctx, dqkv: torch.Tensor, *args, **kwargs):  # type: ignore
                unqueezed_dqkv = dqkv.view([1] + list(dqkv.shape))
                out: tuple = internlm.model.embedding.LegacyApplyRotaryEmbQKV_.backward(
                    ctx, unqueezed_dqkv, *args, **kwargs
                )
                return (out[0].view(out[0].shape[1:]),) + out[1:]

        internlm.model.embedding.apply_rotary_emb_qkv_ = NonLegacyRotaryEmbQKV_.apply

        import builtins
        import internlm.model.norm  # type: ignore

        # HACK: RMSNormTorch class object has been assigned to RMSNorm via
        #           RMSNorm = try_import_RMSNorm()
        #       everywhere (e.g. in modeling_llama.py). Thus simply reassigning
        #       RMSNormTorch to DeepLinkRMSNorm won't work. But we don't want to
        #       reassign every RMSNorm to DeepLinkRMSNorm. So we patch
        #       RMSNormTorch.__new__ to create a DeepLinkRMSNorm instance whenever
        #       RMSNorm(...) is called.
        #       This is not enough though. In latest internevo, there are checks like
        #           if isinstance(module, RMSNorm):
        #       which will fail under this patch. Thus we need also trick `isinstance`.
        internlm.model.norm.RMSNormTorch.__new__ = lambda _, *args, **kwargs: (
            ext.rms_norm.DeepLinkRMSNormWithNormalizedShape(*args, **kwargs)
        )
        isinstance_orig = builtins.isinstance
        builtins.isinstance = lambda obj, class_or_tuple: (
            isinstance_orig(obj, class_or_tuple)
            or (
                (
                    internlm.model.norm.RMSNormTorch
                    in (
                        class_or_tuple
                        if isinstance_orig(class_or_tuple, tuple)
                        else (class_or_tuple,)
                    )
                )
                and isinstance_orig(
                    obj, ext.rms_norm.DeepLinkRMSNormWithNormalizedShape
                )
            )
        )

        import fused_dense_lib  # type: ignore
        import internlm.model.utils  # type: ignore

        fused_dense_lib.linear_bias_wgrad = internlm.model.utils.linear_bias_wgrad_torch

    cpp_ext_found = _find_or_mock_module("deeplink_ext.cpp_extensions")
    if not cpp_ext_found:
        print(
            "[deeplink_ext] WARNING: cpp_extensions not compiled, falling back to pure python implementation"
        )
    _find_or_mock_module("rotary_emb")
    _find_or_mock_module("fused_dense_lib")
    _find_or_mock_module("xentropy_cuda_lib")
    _find_or_mock_module("flash_attn_2_cuda")
    _find_flash_attn()
    if force_fallback:
        _force_fallback()
    _patch_flash_attn()
    _patch_ops()
    print("[deeplink_ext] patched diopi implementation of internlm\n", end="")


_patch_internlm(force_fallback=_force_fallback)
