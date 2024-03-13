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
        import torch_dipu
        if (torch_dipu.dipu.vendor_type == "NPU"):
            from einops import rearrange
            import internlm.model.embedding

            class DeepLinkRotaryEmbedding(internlm.model.embedding.RotaryEmbedding):
                def _update_cos_sin_cache(self, x, indexes):
                    """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
                    if not isinstance(indexes, int):
                        seqlen = indexes.max().item() + 1
                    else:
                        seqlen = indexes + 1  # eval_forward
                    # Reset the tables if the sequence length has changed,
                    # or if we're on a new device (possibly due to tracing for instance)
                    if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
                        self._seq_len_cached = seqlen
                        t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
                        # Don't do einsum, it converts fp32 to fp16
                        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                        if os.environ.get("DEEPLINK_EXT_FORCE_FALLBACK", "0") == "0":
                            freqs = freqs.repeat(1, 2)
                        if self.scale is None:
                            self._cos_cached = torch.cos(freqs).to(x.dtype)
                            self._sin_cached = torch.sin(freqs).to(x.dtype)
                        else:
                            power = (
                                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                            ) / self.scale_base
                            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                            # We want the multiplication by scale to happen in fp32
                            self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                            self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                            self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                            self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

            internlm.model.embedding.RotaryEmbedding = DeepLinkRotaryEmbedding
            setattr(internlm.model.multi_head_attention, "RotaryEmbedding", DeepLinkRotaryEmbedding)

        import internlm.model.modeling_internlm  # type: ignore
        from internlm.model.utils import gather_forward_split_backward
        from internlm.core.context import IS_SEQUENCE_PARALLEL, IS_TENSOR_PARALLEL, ParallelMode
        from internlm.utils.checkpoint import activation_checkpoint

        class DeepLinkPackedFlashBaseLayer1D(internlm.model.modeling_internlm.PackedFlashBaseLayer1D):
            def _forward(self, hidden_states=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None):
                r"""Pass the input through the encoder layer.

                Args:
                    hidden_states: the sequence to the encoder layer (required).
                    residual: hidden_states = Attn/MLP(LN(residual))
                    cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
                    indexes: the length of index is same as hidden states, which stand for the current position
                """
                mixer_kwargs = {
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen,
                    "indexes": indexes,
                    "inference_params": inference_params,
                }

                def _dropout_and_norm_attn(_hidden_states):
                    _dropped = self.dropout1(_hidden_states)
                    _residual = _dropped
                    _hidden_states = self.norm1(_residual)
                    return _residual, _hidden_states

                if self.dropout_selective_checkpoint:
                    residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, hidden_states)
                else:
                    residual, hidden_states = _dropout_and_norm_attn(hidden_states)

                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                hidden_states = self.mixer(hidden_states, **mixer_kwargs)

                def _dropout_and_norm_ffn(_residual, _hidden_states):
                    _dropped = self.dropout2(_hidden_states)
                    _residual = (_dropped + _residual) if _residual is not None else _dropped
                    _hidden_states = self.norm2(_residual)
                    return _residual, _hidden_states

                if self.dropout_selective_checkpoint:
                    residual, hidden_states = activation_checkpoint(_dropout_and_norm_ffn, False, residual, hidden_states)
                else:
                    residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                hidden_states = self.mlp(hidden_states)

                return hidden_states + residual

        internlm.model.modeling_internlm.PackedFlashBaseLayer1D = DeepLinkPackedFlashBaseLayer1D

        class DeepLinkPackedFlashInternLm1D(internlm.model.modeling_internlm.PackedFlashInternLm1D):
            def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
                # attention_mask: compute attention on the places where the value is 1
                if hasattr(self, "embedding"):
                    hidden_states = self.embedding(input_ids)
                    if self.embed_grad_scale != 1:
                        hidden_states = (
                            self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                        )
                if isinstance(cu_seqlens, list):
                    assert len(cu_seqlens) == 1
                    cu_seqlens = cu_seqlens[0].to(hidden_states.device)

                if cu_seqlens is not None:
                    cu_seqlens = cu_seqlens.squeeze(0)
                    hidden_states = hidden_states.squeeze(0)  # If cu_seqlens is passed in，it indicated a packed state，
                    # the batch dimension with a size of 1 should be directly squeezed off.

                if indexes is not None:
                    assert len(indexes) == 1
                    # The indexes are used to indicate the actual position IDs of each token in the packed input.
                    indexes = indexes[0]
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None

                for _, block in enumerate(self.blocks):
                    hidden_states = block(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        indexes=indexes,
                        inference_params=inference_params,
                        max_seqlen=max_seqlen,
                    )

                if hasattr(self, "norm"):
                    hidden_states = self.norm(hidden_states)
                if hasattr(self, "head"):
                    hidden_states = self.head(hidden_states)

                if not self.parallel_output:
                    hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=-1)
                return hidden_states

        internlm.model.modeling_internlm.PackedFlashInternLm1D = DeepLinkPackedFlashInternLm1D

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
