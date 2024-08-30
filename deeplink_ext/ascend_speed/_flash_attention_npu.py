# Copyright (c) 2024, DeepLink.

import torch
import torch_npu

__all__ = ["FlashSelfAttention"]


class FlashSelfAttention(torch.nn.Module):
    """
    Performs self-attention
    """

    def __init__(self, dropout_p=0.0, softmax_scale=None) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale

    def forward(
        self, q, k, v, attention_mask, dropout_p, softmax_scale, head_num, input_layout
    ):
        """
        Performs self-attention on the input sequences.
        """

        seqlen_q = min(q.shape[1], 2048)
        if seqlen_q < 2048:
            sparse_mode = 0
        else:
            sparse_mode = 2

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            input_layout,
            attention_mask=attention_mask,
            scale=softmax_scale,
            keep_prob=1 - dropout_p,
            pre_tockens=seqlen_q,
            next_tockens=0,
            sparse_mode=sparse_mode,
        )[0]

        return out
