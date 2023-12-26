# Copyright (c) 2023, DeepLink.

from typing import Any
import torch
import dipu_ext.ext_
'''
[In]
const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, 
const at::Tensor& queryWeight, const at::Tensor& keyWeight, const at::Tensor& valueWeight, 
const at::Tensor& attnMask, const at::Tensor& outProjWeight,

const c10::optional<at::Tensor>& queryBias, const c10::optional<at::Tensor>& keyBias, const c10::optional<at::Tensor>& valueBias,
const c10::optional<at::Tensor>& outProjBias, 

const c10::optional<at::Tensor>& dropoutMaskInput, const int64_t attnHeadNum,
const int64_t attnHeadDim, const int64_t srcLen, const int64_t tgtLen, const double dropoutProb, const bool softmaxUseFloat
[Ret]
std::move(out), std::move(dropoutMask), std::move(queryRes), std::move(keyRes), std::move(valueRes), std::move(attnScores),
                           std::move(attnRes), std::move(queryContext)

[In]                
const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, 
const at::Tensor& queryWeight, const at::Tensor& keyWeight, const at::Tensor& valueWeight, const at::Tensor& outProjWeight,
const c10::optional<at::Tensor>& queryBias, const c10::optional<at::Tensor>& keyBias, const c10::optional<at::Tensor>& valueBias,
const c10::optional<at::Tensor>& outProjBias, 

const at::Tensor& queryRes, const at::Tensor& keyRes, const at::Tensor& valueRes,
                              const at::Tensor& attnScores, const at::Tensor& attnRes, const at::Tensor& queryContext, const at::Tensor& outGrad,
                              const at::Tensor& dropoutMask, const int64_t attnHeadNum, const int64_t attnHeadDim, const int64_t srcLen, const int64_t tgtLen,
                              const double dropoutProb, const bool softmaxUseFloat
[Res]
    at::Tensor queryWeightGrad = at::empty({weightCol, weightCol}, queryWeight.options());
    at::Tensor keyWeightGrad = at::empty({weightCol, weightCol}, keyWeight.options());
    at::Tensor valueWeightGrad = at::empty({weightCol, weightCol}, valueWeight.options());
    at::Tensor outProjWeightGrad = at::empty({weightCol, weightCol}, outProjWeight.options());
    at::Tensor queryGrad = at::empty({queryShape[0], weightCol}, query.options());
    at::Tensor keyGrad = at::empty({batch * srcLen, weightCol}, key.options());
    at::Tensor valueGrad = at::empty({batch * srcLen, weightCol}, value.options());
    at::Tensor queryBiasGrad = at::empty({1, weightCol}, queryBiasOpt.options());
    at::Tensor keyBiasGrad = at::empty({1, weightCol}, keyBiasOpt.options());
    at::Tensor valueBiasGrad = at::empty({1, weightCol}, valueBiasOpt.options());
    at::Tensor outProjBiasGrad = at::empty({1, weightCol}, outProjBiasOpt.options());
'''


class DeepLinkVanilaMultiHeadAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, wq, wk, wv, attn_mask, w_op, bq, bk, bv, b_op, dropout_mask_in, attn_head_num,
                attn_head_dim, src_len, tgt_len, drop_p, softmax_float) -> Any:
        out, dropout_mask, q_res, k_res, v_res, attn_scores, attn_res, q_ctx = dipu_ext.ext_.multi_head_attn_forward(
            q, k, v, wq, wk, wv, attn_mask, w_op, bq, bk, bv, b_op, dropout_mask_in, attn_head_num, attn_head_dim,
            src_len, tgt_len, drop_p, softmax_float)
        ctx.save_for_backward(
            out,
            dropout_mask,
            q_res,
            k_res,
            v_res,
            attn_scores,
            attn_res,
            q_ctx,
            q,
            k,
            v,
            wq,
            wk,
            wv,
            w_op,
            bq,
            bk,
            bv,
            b_op,
        )
        ctx.attn_head_num = attn_head_num
        ctx.attn_head_dim = attn_head_dim
        ctx.src_len = src_len
        ctx.tgt_len = tgt_len
        ctx.drop_p = drop_p
        ctx.softmax_float = softmax_float

        return out, dropout_mask, attn_scores, attn_res, q_ctx

    @staticmethod
    def backward(ctx) -> Any:
        out, dropout_mask, q_res, k_res, v_res, attn_scores, attn_res, q_ctx, q, k, v, wq, wk, wv, w_op, bq, bk, bv, b_op = ctx.saved_tensors
        qw_grad, kw_grad, vw_grad, opw_grad, q_grad, k_grad, v_grad, qb_grad, kb_grad, vb_grad, opb_grad = dipu_ext.ext_.multi_head_attn_backward(
            q, k, v, wq, wk, wv, w_op, bq, bk, bv, b_op, q_res, k_res, v_res, attn_scores, attn_res, q_ctx, out,
            dropout_mask, ctx.attn_head_num, ctx.attn_head_dim, ctx.src_len, ctx.tgt_len, ctx.drop_p, ctx.softmax_float)
        return qw_grad, kw_grad, vw_grad, opw_grad, q_grad, k_grad, v_grad, qb_grad, kb_grad, vb_grad, opb_grad


class DeepLinkMultiHeadAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)
        out, softmax_lse, rng, S_dmask = dipu_ext.ext_.mha_fwd(
            q,
            k,
            v,
            dropout_p,
            causal,
            return_softmax and dropout_p > 0,
            softmax_scale,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, rng.get_state())
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        rng = torch.Generator(device=q.device)
        rng.set_state(rng_state)
        dq, dk, dv = dipu_ext.ext_.mha_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            ctx.dropout_p,
            ctx.causal,
            rng,
            ctx.softmax_scale,
            None,
            None,
            None,
        )
        return dq, dk, dv, None, None, None, None
