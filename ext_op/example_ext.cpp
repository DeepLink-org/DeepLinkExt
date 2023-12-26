// Copyright (c) 2023, DeepLink.

#include <cstdint>
#include <tuple>
#include <vector>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <torch/csrc/utils/pybind.h>  // IWYU pragma: keep

#include <pybind11/pybind11.h>

#include <diopi/functions_ext.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

#include "diopi_helper.h"
#include "pybind_type_cast.h"

namespace dipu {
namespace dipu_ext {

namespace {

at::IntArrayRef optionalIntArrayToIntArrayRefOrDefault(const OptionalIntArray& opt, at::IntArrayRef def) {
    if (opt) {
        return {*opt};
    }
    return def;
}

}  // namespace

auto extRmsNorm(const at::Tensor& input, const OptionalIntArray& normalized_shape, const at::Tensor& weight, const at::Tensor& bias, double eps) {
    at::OptionalIntArrayRef normalized_shape_at = optionalIntArrayToIntArrayRefOrDefault(normalized_shape, weight.sizes());
    auto inv_rms = at::empty_like(input);
    auto output = at::empty_like(input);
    callDiopi(diopiRMSNorm, output, inv_rms, input, normalized_shape_at, weight, bias, eps);
    return std::make_tuple(std::move(output), std::move(inv_rms));
}

auto extRmsNormBackward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& inv_rms, const OptionalIntArray& normalized_shape,
                        const at::Tensor& weight, const at::Tensor& bias, double eps) {
    at::OptionalIntArrayRef normalized_shape_at = optionalIntArrayToIntArrayRefOrDefault(normalized_shape, weight.sizes());
    auto grad_input = at::empty_like(grad_output);
    auto grad_weight = at::empty_like(weight);
    auto grad_bias = at::empty_like(bias);
    callDiopi(diopiRMSNormBackward, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias, inv_rms, normalized_shape_at, eps);
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

void extApplyRotary(at::Tensor output, const at::Tensor& input, const at::Tensor& cos, const at::Tensor& sin, const bool conj, const bool interleaved = false) {
    callDiopi(diopiRotaryEmbedding, output, input, cos, sin, conj, interleaved);
}

auto extMultiHeadAttention(at::Tensor& q, at::Tensor& k, at::Tensor& v, double dropout_p, bool is_casual, bool return_debug_mask, double scale) {
    const auto batch_size = q.sizes()[0];
    const auto q_seq_len = q.sizes()[1];
    const auto head_num = q.sizes()[2];
    const auto k_seq_len = k.sizes()[1];

    auto out = at::empty_like(q);

    const IntArray softmax_lse_size{batch_size, head_num, q_seq_len};
    const auto softmax_lse_option = q.options().dtype(at::kFloat);
    auto softmax_lse = at::empty(softmax_lse_size, softmax_lse_option);

    auto gen = createDIPUGenerator();

    const auto debug_attn_mask_size = return_debug_mask ? IntArray{batch_size, head_num, q_seq_len, k_seq_len} : IntArray{0};
    const auto debug_attn_mask_option = q.options().dtype(at::kBool);
    auto debug_attn_mask = at::empty(debug_attn_mask_size, debug_attn_mask_option);

    callDiopi(diopiMultiHeadAttention, q, k, v, dropout_p, is_casual, return_debug_mask, scale, out, softmax_lse, gen, debug_attn_mask);
    return std::make_tuple(std::move(out), std::move(softmax_lse), std::move(gen), std::move(debug_attn_mask));
}

// grad_q, grad_k, grad_v are output args, and should be pre-allocated.
auto extMultiHeadAttentionBackward(const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& out,
                                   const at::Tensor& softmax_lse, double dropout_p, bool is_casual, at::Generator& gen, double scale,
                                   c10::optional<at::Tensor>& grad_q_opt, c10::optional<at::Tensor>& grad_k_opt, c10::optional<at::Tensor>& grad_v_opt) {
    auto grad_q = grad_q_opt.has_value() ? grad_q_opt.value() : at::empty_like(q);
    auto grad_k = grad_k_opt.has_value() ? grad_k_opt.value() : at::empty_like(k);
    auto grad_v = grad_v_opt.has_value() ? grad_v_opt.value() : at::empty_like(v);
    callDiopi(diopiMultiHeadAttentionBackward, grad_out, q, k, v, out, softmax_lse, dropout_p, is_casual, gen, scale, grad_q, grad_k, grad_v);
    return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

auto extMultiHeadAttentionVarLen(at::Tensor& q, at::Tensor& k, at::Tensor& v, const at::Tensor& cum_seq_q, const at::Tensor& cum_seq_k, std::int64_t max_q,
                                 std::int64_t max_k, double dropout_p, bool is_casual, bool return_debug_mask, double scale) {
    const auto head_num = q.sizes()[1];
    const auto batch_size = cum_seq_q.sizes()[0] - 1;

    auto out = at::empty_like(q);

    const IntArray softmax_lse_size{batch_size, head_num, max_q};
    const auto softmax_lse_option = q.options().dtype(at::kFloat);
    auto softmax_lse = at::empty(softmax_lse_size, softmax_lse_option);

    auto gen = createDIPUGenerator();

    const auto debug_attn_mask_size = return_debug_mask ? IntArray{batch_size, head_num, max_q, max_k} : IntArray{0};
    const auto debug_attn_mask_option = q.options().dtype(at::kBool);
    auto debug_attn_mask = at::empty(debug_attn_mask_size, debug_attn_mask_option);

    callDiopi(diopiMultiHeadAttentionVarLen, q, k, v, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_casual, return_debug_mask, scale, out, softmax_lse, gen,
              debug_attn_mask);
    return std::make_tuple(std::move(out), std::move(softmax_lse), std::move(gen), std::move(debug_attn_mask));
}

// grad_q, grad_k, grad_v are output args, and should be pre-allocated.
auto extMultiHeadAttentionVarLenBackward(const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& out,
                                         const at::Tensor& softmax_lse, const at::Tensor& cum_seq_q, const at::Tensor& cum_seq_k, std::int64_t max_q,
                                         std::int64_t max_k, double dropout_p, bool is_casual, at::Generator& gen, double scale,
                                         c10::optional<at::Tensor>& grad_q_opt, c10::optional<at::Tensor>& grad_k_opt, c10::optional<at::Tensor>& grad_v_opt) {
    auto grad_q = grad_q_opt.has_value() ? grad_q_opt.value() : at::empty_like(q);
    auto grad_k = grad_k_opt.has_value() ? grad_k_opt.value() : at::empty_like(k);
    auto grad_v = grad_v_opt.has_value() ? grad_v_opt.value() : at::empty_like(v);
    callDiopi(diopiMultiHeadAttentionVarLenBackward, grad_out, q, k, v, out, softmax_lse, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_casual, gen, scale,
              grad_q, grad_k, grad_v);
    return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

auto extDestIndexCopyKV(const at::Tensor& k, const at::Tensor& dest_loc, at::Tensor& out) {
    callDiopi(diopiDestIndexCopyKV, out, k, dest_loc);
    return out;
}

auto extTokenAttentionInference(const at::Tensor& q, const at::Tensor& k, at::Tensor& out, const at::Tensor& b_loc, const at::Tensor& b_start_loc,
                                const at::Tensor& b_seq_len, int max_input_len) {
    callDiopi(diopiTokenAttentionInference, out, q, k, b_loc, b_start_loc, b_seq_len, max_input_len);
    return out;
}

auto extTokenSoftmaxReduceVInference(const at::Tensor& logics, const at::Tensor& v, at::Tensor& out, const at::Tensor& b_loc, const at::Tensor& b_start_loc,
                                     const at::Tensor& b_seq_len, int max_input_len, int other_kv_index) {
    callDiopi(diopiTokenSoftmaxReduceVInference, out, logics, v, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index);
    return out;
}

auto extContextAttentionInference(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, const at::Tensor& b_start_loc,
                                  const at::Tensor& b_seq_len, int max_input_len) {
    callDiopi(diopiContextAttentionInference, out, q, k, v, b_start_loc, b_seq_len, max_input_len);
    return out;
}

at::Tensor extApplyPenalty(at::Tensor& Logits, const at::Tensor& presence_penalty, const at::Tensor& frequency_penalty, const at::Tensor& p_token_ids,
                           const at::Tensor& p_token_counts, const at::Tensor& p_cumsum_seq_len, int p_max_len_in_batch) {
    callDiopi(diopiApplyPenalty, Logits, presence_penalty, frequency_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch);
    return Logits;
}

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
auto extMultiHeadAttnForward(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& queryWeight,
                             const at::Tensor& keyWeight, const at::Tensor& valueWeight, const at::Tensor& attnMask, const at::Tensor& outProjWeight,
                             const c10::optional<at::Tensor>& queryBias, const c10::optional<at::Tensor>& keyBias, const c10::optional<at::Tensor>& valueBias,
                             const c10::optional<at::Tensor>& outProjBias, const c10::optional<at::Tensor>& dropoutMaskInput, const int64_t attnHeadNum,
                             const int64_t attnHeadDim, const int64_t srcLen, const int64_t tgtLen, const double dropoutProb, const bool softmaxUseFloat) {
    const at::Tensor& queryBiasOpt = c10::value_or_else(queryBias, [] { return at::Tensor(); });
    const at::Tensor& keyBiasOpt = c10::value_or_else(keyBias, [] { return at::Tensor(); });
    const at::Tensor& valueBiasOpt = c10::value_or_else(valueBias, [] { return at::Tensor(); });
    const at::Tensor& outProjBiasOpt = c10::value_or_else(outProjBias, [] { return at::Tensor(); });
    const at::Tensor& dropoutMaskInputOpt = c10::value_or_else(dropoutMaskInput, [] { return at::Tensor(); });

    auto queryShape = query.sizes();
    int64_t batch = queryShape[0] / tgtLen;
    auto weightCol = attnHeadNum * attnHeadDim;

    auto queryOptions = query.options();

    at::Tensor out = at::empty({queryShape[0], weightCol}, queryOptions);
    at::Tensor dropoutMask = at::empty({batch * attnHeadNum * tgtLen * srcLen / 8}, queryOptions.dtype(at::kByte));
    at::Tensor queryRes = at::empty({batch, attnHeadNum, tgtLen, attnHeadDim}, queryOptions);
    at::Tensor keyRes = at::empty({batch, attnHeadNum, srcLen, attnHeadDim}, queryOptions);
    at::Tensor valueRes = at::empty({batch, attnHeadNum, srcLen, attnHeadDim}, queryOptions);
    at::Tensor attnScores;
    if (softmaxUseFloat) {
        attnScores = at::empty({batch, attnHeadNum, tgtLen, srcLen}, query.options().dtype(at::kFloat));
    } else {
        attnScores = at::empty({batch, attnHeadNum, tgtLen, srcLen}, queryOptions);
    }
    at::Tensor attnRes = at::empty({batch, attnHeadNum, tgtLen, srcLen}, queryOptions);
    at::Tensor queryContext = at::empty({queryShape[0], weightCol}, queryOptions);
    callDiopi(diopiMultiHeadAttnForward, out, dropoutMask, queryRes, keyRes, valueRes, attnScores, attnRes, queryContext, query, key, value, queryWeight,
              keyWeight, valueWeight, attnMask, outProjWeight, queryBiasOpt, keyBiasOpt, valueBiasOpt, outProjBiasOpt, dropoutMaskInputOpt, attnHeadNum,
              attnHeadDim, srcLen, tgtLen, dropoutProb, softmaxUseFloat);
    return std::make_tuple(std::move(out), std::move(dropoutMask), std::move(queryRes), std::move(keyRes), std::move(valueRes), std::move(attnScores),
                           std::move(attnRes), std::move(queryContext));
}

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
auto extMultiHeadAttnBackward(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& queryWeight,
                              const at::Tensor& keyWeight, const at::Tensor& valueWeight, const at::Tensor& outProjWeight,
                              const c10::optional<at::Tensor>& queryBias, const c10::optional<at::Tensor>& keyBias, const c10::optional<at::Tensor>& valueBias,
                              const c10::optional<at::Tensor>& outProjBias, const at::Tensor& queryRes, const at::Tensor& keyRes, const at::Tensor& valueRes,
                              const at::Tensor& attnScores, const at::Tensor& attnRes, const at::Tensor& queryContext, const at::Tensor& outGrad,
                              const at::Tensor& dropoutMask, const int64_t attnHeadNum, const int64_t attnHeadDim, const int64_t srcLen, const int64_t tgtLen,
                              const double dropoutProb, const bool softmaxUseFloat) {
    const at::Tensor& queryBiasOpt = c10::value_or_else(queryBias, [] { return at::Tensor(); });
    const at::Tensor& keyBiasOpt = c10::value_or_else(keyBias, [] { return at::Tensor(); });
    const at::Tensor& valueBiasOpt = c10::value_or_else(valueBias, [] { return at::Tensor(); });
    const at::Tensor& outProjBiasOpt = c10::value_or_else(outProjBias, [] { return at::Tensor(); });

    auto queryShape = query.sizes();
    int64_t batch = queryShape[0] / tgtLen;
    auto weightCol = attnHeadNum * attnHeadDim;

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

    callDiopi(diopiMultiHeadAttnBackward, queryWeightGrad, keyWeightGrad, valueWeightGrad, outProjWeightGrad, queryGrad, keyGrad, valueGrad, queryBiasGrad,
              keyBiasGrad, valueBiasGrad, outProjBiasGrad, query, key, value, queryWeight, keyWeight, valueWeight, outProjWeight, queryBiasOpt, keyBiasOpt,
              valueBiasOpt, outProjBiasOpt, queryRes, keyRes, valueRes, attnScores, attnRes, queryContext, outGrad, dropoutMask, attnHeadNum, attnHeadDim,
              srcLen, tgtLen, dropoutProb, softmaxUseFloat);
    return std::make_tuple(std::move(queryWeightGrad), std::move(keyWeightGrad), std::move(valueWeightGrad), std::move(outProjWeightGrad), std::move(queryGrad),
                           std::move(keyGrad), std::move(valueGrad), std::move(queryBiasGrad), std::move(keyBiasGrad), std::move(valueBiasGrad),
                           std::move(outProjBiasGrad));
}
/**
 * 判断是否有对应的 diopi 实现:
 * 如果有, 则直接 pybind 上去;
 * 否则不注册, 等到 python 层处理.
*/
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    if (&diopiRMSNorm != nullptr) {  // Check if weak symbol defined
        m.def("rms_norm", &extRmsNorm, "deeplink ext_rms_norm");
    }
    if (&diopiRMSNormBackward != nullptr) {
        m.def("rms_norm_backward", &extRmsNormBackward, "deeplink ext_rms_norm_backward");
    }
    if (&diopiRotaryEmbedding != nullptr) {
        m.def("apply_rotary", &extApplyRotary, "deeplink ext_apply_rotary");
    }
    if (&diopiMultiHeadAttention != nullptr) {
        m.def("mha_fwd", &extMultiHeadAttention, "deeplink ext_mha_fwd");
    }
    if (&diopiMultiHeadAttentionBackward != nullptr) {
        m.def("mha_bwd", &extMultiHeadAttentionBackward, "deeplink ext_mha_bwd");
    }
    if (&diopiMultiHeadAttentionVarLen != nullptr) {
        m.def("mha_varlen_fwd", &extMultiHeadAttentionVarLen, "deeplink ext_mha_varlen_fwd");
    }
    if (&diopiMultiHeadAttentionVarLenBackward != nullptr) {
        m.def("mha_varlen_bwd", &extMultiHeadAttentionVarLenBackward, "deeplink ext_mha_varlen_bwd");
    }
    if (&diopiDestIndexCopyKV != nullptr) {
        m.def("dest_index_copy_kv", &extDestIndexCopyKV, "deeplink ext_dest_index_copy_kv");
    }
    if (&diopiTokenAttentionInference != nullptr) {
        m.def("token_attention_inference", &extTokenAttentionInference, "deeplink ext_token_attention_inference");
    }
    if (&diopiTokenSoftmaxReduceVInference != nullptr) {
        m.def("token_softmax_reducev_inference", &extTokenSoftmaxReduceVInference, "deeplink ext_token_softmax_reducev_inference");
    }
    if (&diopiContextAttentionInference != nullptr) {
        m.def("context_attention_inference", &extContextAttentionInference, "deeplink ext_context_attention_inference");
    }
    if (&diopiApplyPenalty != nullptr) {
        m.def("apply_penalty", &extApplyPenalty, "deeplink ext_apply_penalty");
    }
    if (&extMultiHeadAttnForward != nullptr) {
        m.def("multi_head_attn_forward", &extMultiHeadAttnForward, "deeplink multi_head_attn_forward");
    }
    if (&extMultiHeadAttnBackward != nullptr) {
        m.def("multi_head_attn_backward", &extMultiHeadAttnBackward, "deeplink multi_head_attn_backward");
    }
}

}  // namespace dipu_ext
}  // namespace dipu
