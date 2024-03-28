# Copyright (c) 2024, DeepLink.

import os

__all__ = []

_force_fallback = os.environ.get("DEEPLINK_EXT_FORCE_FALLBACK", "0") != "0"


def _patch_internlm(force_fallback: bool = False):
    import importlib.util
    import sys
    import types
    import torch

    torch.cuda.set_device(int(os.environ["RANK"]) % torch.cuda.device_count())
    print(int(os.environ["RANK"]) % torch.cuda.device_count())
    print(f"torch.cuda.cur_device:{torch.cuda.current_device()}, totaldevice:{torch.cuda.device_count()}")
    import time
    time.sleep(3)

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
        if not force_fallback:
            from einops import rearrange
            import deeplink_ext.cpp_extensions as cpp_ext

            class DeeplinkApplyRotaryEmb(torch.autograd.Function):
                """
                ApplyRotaryEmb
                """

                @staticmethod
                def forward(ctx, x, cos, sin, interleaved=False):
                    """
                        x: (batch_size, seqlen, nheads, headdim)
                        cos, sin: (seqlen, rotary_dim / 2)
                        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                            of 1st half and 2nd half (GPT-NeoX style).
                    rotary_dim must be <= headdim
                    Apply rotary embedding to the first rotary_dim of x.
                    """
                    _, seqlen, _, headdim = x.shape
                    rotary_seqlen, rotary_dim = cos.shape
                    rotary_dim *= 2
                    assert rotary_dim <= headdim
                    assert seqlen <= rotary_seqlen
                    assert sin.shape == (rotary_seqlen, rotary_dim // 2)
                    out = torch.empty_like(x)
                    new_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
                    new_sin = rearrange(sin[:seqlen], "s d -> s 1 d")
                    if rotary_dim < headdim:
                        out[..., rotary_dim:].copy_(x[..., rotary_dim:])
                    ctx.save_for_backward(new_cos, new_sin)
                    ctx.interleaved = interleaved
                    cpp_ext.apply_rotary(
                        out[..., :rotary_dim],
                        x[..., :rotary_dim],
                        new_cos,
                        new_sin,
                        False,
                        interleaved,
                    )
                    return out

                @staticmethod
                def backward(ctx, do):
                    cos, sin = ctx.saved_tensors
                    _, seqlen, _, headdim = do.shape
                    rotary_dim = cos.shape[-1]
                    rotary_dim *= 2
                    dx = torch.empty_like(do)
                    cpp_ext.apply_rotary(
                        dx[..., :rotary_dim],
                        do[..., :rotary_dim],
                        cos,
                        sin,
                        True,
                        ctx.interleaved,
                    )
                    if rotary_dim < headdim:
                        dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
                    return dx, None, None, None, None
            import internlm.model.modules.embedding
            internlm.model.modules.embedding.apply_rotary_emb = DeeplinkApplyRotaryEmb.apply

            class DeeplinkApplyRotaryEmbQKV_(torch.autograd.Function):
                """
                ApplyRotaryEmbQKV_
                """

                @staticmethod
                def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
                    """
                        qkv: (total, 3, nheads, headdim) / (batch_size, seqlen, 3, nheads, headdim)
                        cos, sin: (seqlen, rotary_dim / 2)
                        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
                        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                            1st half and 2nd half (GPT-NeoX style).
                    rotary_dim must be <= headdim
                    Apply rotary embedding *inplace* to the first rotary_dim of q and k.
                    """
                    # len(qkv.shape) == 4 means the format of qkv is (total, 3, nheads, headdim) which is packed,
                    # otherwise the format of qkv is (batch_size, seqlen, 3, nheads, headdim) which is unpacked.
                    # We handle both packed qkv and unpacked qkv scenario in this class.
                    three = qkv.shape[1] if len(qkv.shape) == 4 else qkv.shape[2]
                    assert three == 3
                    seqlen = None if len(qkv.shape) == 4 else qkv.shape[1]
                    rotary_seqlen, rotary_dim = cos.shape
                    if len(qkv.shape) != 4:
                        assert seqlen <= rotary_seqlen
                    headdim = qkv.shape[-1]
                    rotary_dim *= 2
                    assert rotary_dim <= headdim
                    cos_k = cos if cos_k is None else cos_k
                    sin_k = sin if sin_k is None else sin_k
                    assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
                    q_ro = qkv[:, 0, :, :rotary_dim] if len(qkv.shape) == 4 else qkv[:, :, 0, :, :rotary_dim]
                    re_cos = rearrange(cos, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(cos[:seqlen], "s d -> s 1 d")
                    re_sin = rearrange(sin, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(sin[:seqlen], "s d -> s 1 d")
                    out = torch.empty_like(q_ro)
                    cpp_ext.apply_rotary(
                        out,
                        q_ro,
                        re_cos,
                        re_sin,
                        False,
                        interleaved,
                    )
                    q_ro.copy_(out)

                    k_ro = qkv[:, 1, :, :rotary_dim] if len(qkv.shape) == 4 else qkv[:, :, 1, :, :rotary_dim]
                    re_cos_k = (
                        rearrange(cos_k, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(cos_k[:seqlen], "s d -> s 1 d")
                    )
                    re_sin_k = (
                        rearrange(sin_k, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(sin_k[:seqlen], "s d -> s 1 d")
                    )
                    out = torch.empty_like(k_ro)
                    cpp_ext.apply_rotary(
                        out,
                        k_ro,
                        re_cos_k,
                        re_sin_k,
                        False,
                        interleaved,
                    )
                    k_ro.copy_(out)
                    ctx.save_for_backward(re_cos, re_sin, re_cos_k, re_sin_k)
                    ctx.interleaved = interleaved
                    return qkv

                @staticmethod
                def backward(ctx, dqkv):
                    interleaved = ctx.interleaved
                    cos, sin, cos_k, sin_k = ctx.saved_tensors
                    seqlen = None if len(dqkv.shape) == 4 else dqkv.shape[1]
                    rotary_dim = cos.shape[-1]
                    rotary_dim *= 2
                    dq_ro = dqkv[:, 0, :, :rotary_dim] if len(dqkv.shape) == 4 else dqkv[:, :, 0, :, :rotary_dim]
                    out = torch.empty_like(dq_ro)
                    cpp_ext.apply_rotary(
                        out,
                        dq_ro,
                        cos,
                        sin,
                        True,
                        interleaved,
                    )
                    dq_ro.copy_(out)

                    dk_ro = dqkv[:, 1, :, :rotary_dim] if len(dqkv.shape) == 4 else dqkv[:, :, 1, :, :rotary_dim]
                    out = torch.empty_like(dk_ro)
                    cpp_ext.apply_rotary(
                        out,
                        dk_ro,
                        cos_k,
                        sin_k,
                        True,
                        interleaved,
                    )
                    dk_ro.copy_(out)
                    return dqkv, None, None, None, None, None
            internlm.model.modules.apply_rotary_emb_qkv_ = DeeplinkApplyRotaryEmbQKV_.apply

        import deeplink_ext.internlm_ops as ext
        import builtins
        import internlm.model.ops.norm  # type: ignore

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
        internlm.model.ops.norm.RMSNormTorch.__new__ = lambda _, *args, **kwargs: (
            ext.rms_norm.DeepLinkRMSNormWithNormalizedShape(*args, **kwargs)
        )
        isinstance_orig = builtins.isinstance
        builtins.isinstance = lambda obj, class_or_tuple: (
            isinstance_orig(obj, class_or_tuple)
            or (
                (
                    internlm.model.ops.norm.RMSNormTorch
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
