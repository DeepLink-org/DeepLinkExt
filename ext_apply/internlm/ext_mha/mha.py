# Copyright (c) 2023, DeepLink.

import torch.nn as nn
from .mha_qkvpacked_func import DeepLinkMultiHeadAttentionQKVPackedFunc
from .mha_varlen_qkvpacked_func import DeepLinkMultiHeadAttentionVarLenQKVPackedFunc
from .mha_kvpacked_func import DeepLinkMultiHeadAttentionKVPackedFunc
from .mha_varlen_kvpacked_func import DeepLinkMultiHeadAttentionVarLenKVPackedFunc


class DeepLinkSelfAttention(nn.Module):
    """Performs self-attention with support for both padded and unpadded sequences.

    Args:
        causal (bool, optional): If True, applies causal self-attention, meaning each
            position can only attend to previous positions. Default is False.
        softmax_scale (float, optional): Scaling factor applied to the softmax
            operation. If not provided, will be D^{-0.5}. Default is None.
        dropout_p (float, optional): Dropout probability applied to the attention
            scores. Default is 0.0.
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        """Performs self-attention on the input sequences.

        Args:
            qkv (torch.Tensor): Input tensor representing queries, keys, and values
                concatenated together. (B, S, 3, H, D) for padded; (total, 3, H, D)
                for unpadded.
            causal (bool, optional): If provided, overrides the class-level 'causal'
                argument for this forward pass. Default is None.
            cu_seqlens (torch.Tensor((batch_size + 1,), dtype=torch.int32), optional):
                Sequence lengths tensor for unpadded sequences. If provided, performs
                attention on unpadded sequences. Default is None.
            max_seqlen (int, optional): Maximum sequence length for unpadded sequences.
                If provided, defines the maximum length of the sequences. Default is
                None.

        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        if cu_seqlens is None:
            # padded
            return DeepLinkMultiHeadAttentionQKVPackedFunc.apply(
                qkv,
                self.dropout_p,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )
        else:
            # unpadded
            return DeepLinkMultiHeadAttentionVarLenQKVPackedFunc.apply(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.dropout_p,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )


class DeepLinkCrossAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, dropout_p=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout_p

    def forward(
        self,
        q,
        kv,
        causal=None,
        cu_seqlens_q=None,
        max_seqlen_q=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        if cu_seqlens_q is None:
            # padded
            return DeepLinkMultiHeadAttentionKVPackedFunc.apply(
                q,
                kv,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )
        else:
            # unpadded
            return DeepLinkMultiHeadAttentionVarLenKVPackedFunc.apply(
                q,
                kv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout_p if self.training else 0.0,
                self.softmax_scale,
                causal if causal is not None else self.causal,
                False,
            )
