// Copyright (c) 2023, DeepLink.

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <torch/csrc/utils/pybind.h>  // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>

#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

#include "diopi_helper.h"
#include "pybind_type_cast.h"

namespace dipu::dipu_ext {

void extAdamW(at::Tensor& param, at::Tensor& exp_avg, at::Tensor& exp_avg_sq,
              c10::optional<at::Tensor>& max_exp_avg_sq_opt, at::Tensor& grad,
              float lr, float beta1, float beta2, float epsilon,
              float weight_decay, int64_t step, bool amsgrad) {
  // the diopiAdamW func has no "maximize" param
  callDiopi(diopiAdamW, param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq_opt,
            lr, beta1, beta2, epsilon, weight_decay, step, amsgrad);
}

void extRmsNorm(at::Tensor& output, at::Tensor& inv_rms,
                const at::Tensor& input,
                const at::IntArrayRef& normalized_shape,
                const at::Tensor& weight,
                const c10::optional<at::Tensor>& bias_opt, double eps) {
  callDiopi(diopiRMSNorm, output, inv_rms, input, normalized_shape, weight,
            bias_opt, eps);
}

void extRmsNormBackward(at::Tensor& grad_input, at::Tensor& grad_weight,
                        c10::optional<at::Tensor>& grad_bias_opt,
                        const at::Tensor& grad_output, const at::Tensor& input,
                        const at::Tensor& weight,
                        const c10::optional<at::Tensor>& bias_opt,
                        const at::Tensor& inv_rms,
                        const at::IntArrayRef& normalized_shape, double eps) {
  callDiopi(diopiRMSNormBackward, grad_input, grad_weight, grad_bias_opt,
            grad_output, input, weight, bias_opt, inv_rms, normalized_shape,
            eps);
}

void extApplyRotary(at::Tensor& output, const at::Tensor& input,
                    const at::Tensor& cos, const at::Tensor& sin,
                    const bool conj, const bool interleaved) {
  callDiopi(diopiRotaryEmbedding, output, input, cos, sin, conj, interleaved);
}

auto extMultiHeadAttention(at::Tensor& q, at::Tensor& k, at::Tensor& v,
                           double dropout_p, bool is_causal,
                           bool return_debug_mask, double scale) {
  const auto batch_size = q.sizes()[0];
  const auto q_seq_len = q.sizes()[1];
  const auto head_num = q.sizes()[2];
  const auto k_seq_len = k.sizes()[1];

  auto out = at::empty_like(q);

  const IntArray softmax_lse_size{batch_size, head_num, q_seq_len};
  const auto softmax_lse_option = q.options().dtype(at::kFloat);
  auto softmax_lse = at::empty(softmax_lse_size, softmax_lse_option);

  auto gen = createDIPUGenerator();

  const auto debug_attn_mask_size =
      return_debug_mask ? IntArray{batch_size, head_num, q_seq_len, k_seq_len}
                        : IntArray{0};
  const auto debug_attn_mask_option = q.options().dtype(at::kBool);
  auto debug_attn_mask =
      at::empty(debug_attn_mask_size, debug_attn_mask_option);

  callDiopi(diopiMultiHeadAttention, q, k, v, dropout_p, is_causal,
            return_debug_mask, scale, out, softmax_lse, gen, debug_attn_mask);
  return std::make_tuple(std::move(out), std::move(softmax_lse), std::move(gen),
                         std::move(debug_attn_mask));
}

// grad_q, grad_k, grad_v are output args, and should be pre-allocated.
auto extMultiHeadAttentionBackward(const at::Tensor& grad_out,
                                   const at::Tensor& q, const at::Tensor& k,
                                   const at::Tensor& v, const at::Tensor& out,
                                   const at::Tensor& softmax_lse,
                                   double dropout_p, bool is_causal,
                                   at::Generator& gen, double scale,
                                   c10::optional<at::Tensor>& grad_q_opt,
                                   c10::optional<at::Tensor>& grad_k_opt,
                                   c10::optional<at::Tensor>& grad_v_opt) {
  auto grad_q = grad_q_opt.has_value() ? grad_q_opt.value() : at::empty_like(q);
  auto grad_k = grad_k_opt.has_value() ? grad_k_opt.value() : at::empty_like(k);
  auto grad_v = grad_v_opt.has_value() ? grad_v_opt.value() : at::empty_like(v);
  callDiopi(diopiMultiHeadAttentionBackward, grad_out, q, k, v, out,
            softmax_lse, dropout_p, is_causal, gen, scale, grad_q, grad_k,
            grad_v);
  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v));
}

auto extMultiHeadAttentionVarLen(at::Tensor& q, at::Tensor& k, at::Tensor& v,
                                 const at::Tensor& cum_seq_q,
                                 const at::Tensor& cum_seq_k,
                                 std::int64_t max_q, std::int64_t max_k,
                                 double dropout_p, bool is_causal,
                                 bool return_debug_mask, double scale) {
  const auto head_num = q.sizes()[1];
  const auto batch_size = cum_seq_q.sizes()[0] - 1;

  auto out = at::empty_like(q);

  const IntArray softmax_lse_size{batch_size, head_num, max_q};
  const auto softmax_lse_option = q.options().dtype(at::kFloat);
  auto softmax_lse = at::empty(softmax_lse_size, softmax_lse_option);

  auto gen = createDIPUGenerator();

  const auto debug_attn_mask_size =
      return_debug_mask ? IntArray{batch_size, head_num, max_q, max_k}
                        : IntArray{0};
  const auto debug_attn_mask_option = q.options().dtype(at::kBool);
  auto debug_attn_mask =
      at::empty(debug_attn_mask_size, debug_attn_mask_option);

  callDiopi(diopiMultiHeadAttentionVarLen, q, k, v, cum_seq_q, cum_seq_k, max_q,
            max_k, dropout_p, is_causal, return_debug_mask, scale, out,
            softmax_lse, gen, debug_attn_mask);
  return std::make_tuple(std::move(out), std::move(softmax_lse), std::move(gen),
                         std::move(debug_attn_mask));
}

// grad_q, grad_k, grad_v are output args, and should be pre-allocated.
auto extMultiHeadAttentionVarLenBackward(
    const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& out, const at::Tensor& softmax_lse,
    const at::Tensor& cum_seq_q, const at::Tensor& cum_seq_k,
    std::int64_t max_q, std::int64_t max_k, double dropout_p, bool is_causal,
    at::Generator& gen, double scale, c10::optional<at::Tensor>& grad_q_opt,
    c10::optional<at::Tensor>& grad_k_opt,
    c10::optional<at::Tensor>& grad_v_opt) {
  auto grad_q = grad_q_opt.has_value() ? grad_q_opt.value() : at::empty_like(q);
  auto grad_k = grad_k_opt.has_value() ? grad_k_opt.value() : at::empty_like(k);
  auto grad_v = grad_v_opt.has_value() ? grad_v_opt.value() : at::empty_like(v);
  callDiopi(diopiMultiHeadAttentionVarLenBackward, grad_out, q, k, v, out,
            softmax_lse, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
            is_causal, gen, scale, grad_q, grad_k, grad_v);
  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v));
}

void extFlashAttention(at::Tensor& out, at::Tensor& softmax_lse,
                       at::Generator& gen, const at::Tensor& q,
                       const at::Tensor& k, const at::Tensor& v,
                       const c10::optional<at::Tensor>& alibi_slopes_opt,
                       float p_dropout, float softmax_scale, bool is_causal,
                       int32_t window_size_left, int32_t window_size_right) {
  callDiopi(diopiFlashAttention, out, softmax_lse, gen, q, k, v,
            alibi_slopes_opt, p_dropout, softmax_scale, is_causal,
            window_size_left, window_size_right);
}

void extFlashAttentionBackward(
    at::Tensor& grad_q, at::Tensor& grad_k, at::Tensor& grad_v,
    const at::Tensor& grad_out, at::Generator& gen, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& alibi_slopes_opt, const at::Tensor& out,
    const at::Tensor& softmax_lse, float p_dropout, float softmax_scale,
    bool is_causal, int32_t window_size_left, int32_t window_size_right) {
  callDiopi(diopiFlashAttentionBackward, grad_q, grad_k, grad_v, grad_out, gen,
            q, k, v, alibi_slopes_opt, out, softmax_lse, p_dropout,
            softmax_scale, is_causal, window_size_left, window_size_right);
}

void extFlashAttentionVarLen(
    at::Tensor& out, at::Tensor& softmax_lse, at::Generator& gen,
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& cum_seq_q, const at::Tensor& cum_seq_kv,
    const c10::optional<at::Tensor>& alibi_slopes_opt, int32_t max_seqlen_q,
    int32_t max_seqlen_kv, float p_dropout, float softmax_scale, bool is_causal,
    int32_t window_size_left, int32_t window_size_right) {
  callDiopi(diopiFlashAttentionVarLen, out, softmax_lse, gen, q, k, v,
            cum_seq_q, cum_seq_kv, alibi_slopes_opt, max_seqlen_q,
            max_seqlen_kv, p_dropout, softmax_scale, is_causal,
            window_size_left, window_size_right);
}

void extFlashAttentionVarLenBackward(
    at::Tensor& grad_q, at::Tensor& grad_k, at::Tensor& grad_v,
    const at::Tensor& grad_out, at::Generator& gen, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v, const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_kv,
    const c10::optional<at::Tensor>& alibi_slopes_opt, const at::Tensor& out,
    const at::Tensor& softmax_lse, int32_t max_seqlen_q, int32_t max_seqlen_kv,
    float p_dropout, float softmax_scale, bool is_causal,
    int32_t window_size_left, int32_t window_size_right) {
  callDiopi(diopiFlashAttentionVarLenBackward, grad_q, grad_k, grad_v, grad_out,
            gen, q, k, v, cum_seq_q, cum_seq_kv, alibi_slopes_opt, out,
            softmax_lse, max_seqlen_q, max_seqlen_kv, p_dropout, softmax_scale,
            is_causal, window_size_left, window_size_right);
}

// for ascend
auto extCustomizedFlashAttention(
    at::Tensor& out, at::Generator& gen, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& alibi_slopes_opt,
    const c10::optional<at::Tensor>& attention_mask_opt, float p_dropout,
    float softmax_scale, bool is_causal, int32_t window_size_left,
    int32_t window_size_right) {
  diopiTensorHandle_t dropout_mask = nullptr;
  diopiTensorHandle_t softmax_max = nullptr;
  diopiTensorHandle_t softmax_sum = nullptr;
  diopiTensorHandle_t softmax_out = nullptr;

  [[maybe_unused]] auto context = callDiopiKeepContext(
      diopiCustomizedFlashAttention, out, &dropout_mask, &softmax_max,
      &softmax_sum, &softmax_out, gen, q, k, v, alibi_slopes_opt,
      attention_mask_opt, p_dropout, softmax_scale, is_causal, window_size_left,
      window_size_right);

  return std::make_tuple(
      dropout_mask ? *dipu::diopi_helper::fromDiopiTensorHandle(dropout_mask)
                   : at::Tensor(),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_max),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_sum),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_out));
}

// for ascend
void extCustomizedFlashAttentionBackward(
    at::Tensor& grad_q, at::Tensor& grad_k, at::Tensor& grad_v,
    const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const c10::optional<at::Tensor>& alibi_slopes_opt,
    const at::Tensor& out, const c10::optional<at::Tensor>& attention_mask_opt,
    const c10::optional<at::Tensor>& dropout_mask_opt,
    const at::Tensor& softmax_max, const at::Tensor& softmax_sum,
    const at::Tensor& softmax_out, float p_dropout, float softmax_scale,
    bool is_causal, int32_t window_size_left, int32_t window_size_right) {
  callDiopi(diopiCustomizedFlashAttentionBackward, grad_q, grad_k, grad_v,
            grad_out, q, k, v, alibi_slopes_opt, out, attention_mask_opt,
            dropout_mask_opt, softmax_max, softmax_sum, softmax_out, p_dropout,
            softmax_scale, is_causal, window_size_left, window_size_right);
}

// for ascend
auto extCustomizedFlashAttentionVarLen(
    at::Tensor& out, at::Generator& gen, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v, const at::IntArrayRef& cum_seq_q,
    const at::IntArrayRef& cum_seq_kv,
    const c10::optional<at::Tensor>& alibi_slopes_opt,
    const c10::optional<at::Tensor>& attention_mask_opt, int32_t max_seqlen_q,
    int32_t max_seqlen_kv, float p_dropout, float softmax_scale, bool is_causal,
    int32_t window_size_left, int32_t window_size_right) {
  diopiTensorHandle_t dropout_mask = nullptr;
  diopiTensorHandle_t softmax_max = nullptr;
  diopiTensorHandle_t softmax_sum = nullptr;
  diopiTensorHandle_t softmax_out = nullptr;

  [[maybe_unused]] auto context = callDiopiKeepContext(
      diopiCustomizedFlashAttentionVarLen, out, &dropout_mask, &softmax_max,
      &softmax_sum, &softmax_out, gen, q, k, v, cum_seq_q, cum_seq_kv,
      alibi_slopes_opt, attention_mask_opt, max_seqlen_q, max_seqlen_kv,
      p_dropout, softmax_scale, is_causal, window_size_left, window_size_right);

  return std::make_tuple(
      dropout_mask ? *dipu::diopi_helper::fromDiopiTensorHandle(dropout_mask)
                   : at::Tensor(),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_max),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_sum),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_out));
}

// for ascend
void extCustomizedFlashAttentionVarLenBackward(
    at::Tensor& grad_q, at::Tensor& grad_k, at::Tensor& grad_v,
    const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::IntArrayRef& cum_seq_q,
    const at::IntArrayRef& cum_seq_kv,
    const c10::optional<at::Tensor>& alibi_slopes_opt, const at::Tensor& out,
    const c10::optional<at::Tensor>& attention_mask_opt,
    const c10::optional<at::Tensor>& dropout_mask,
    const at::Tensor& softmax_max, const at::Tensor& softmax_sum,
    const at::Tensor& softmax_out, int32_t max_seqlen_q, int32_t max_seqlen_kv,
    float p_dropout, float softmax_scale, bool is_causal,
    int32_t window_size_left, int32_t window_size_right) {
  callDiopi(diopiCustomizedFlashAttentionVarLenBackward, grad_q, grad_k, grad_v,
            grad_out, q, k, v, cum_seq_q, cum_seq_kv, alibi_slopes_opt, out,
            attention_mask_opt, dropout_mask, softmax_max, softmax_sum,
            softmax_out, max_seqlen_q, max_seqlen_kv, p_dropout, softmax_scale,
            is_causal, window_size_left, window_size_right);
}

void extScaledMaskedSoftmax(at::Tensor& out, const at::Tensor& input,
                            const at::Tensor& mask, double scale,
                            bool fixed_triu_mask) {
  callDiopi(diopiScaledMaskedSoftmax, out, input, mask, scale, fixed_triu_mask);
}

void extScaledMaskedSoftmaxBackward(at::Tensor& grad_input,
                                    const at::Tensor& grad_output,
                                    const at::Tensor& out,
                                    const at::Tensor& mask, double scale,
                                    bool fixed_triu_mask) {
  callDiopi(diopiScaledMaskedSoftmaxBackward, grad_input, grad_output, out,
            mask, scale, fixed_triu_mask);
}

void extDestIndexCopyKV(const at::Tensor& k, const at::Tensor& dest_loc,
                        at::Tensor& out) {
  callDiopi(diopiDestIndexCopyKV, out, k, dest_loc);
}

void extTokenAttentionInference(const at::Tensor& q, const at::Tensor& k,
                                at::Tensor& out, const at::Tensor& b_loc,
                                const at::Tensor& b_start_loc,
                                const at::Tensor& b_seq_len,
                                int max_input_len) {
  callDiopi(diopiTokenAttentionInference, out, q, k, b_loc, b_start_loc,
            b_seq_len, max_input_len);
}

void extTokenSoftmaxReduceVInference(const at::Tensor& logics,
                                     const at::Tensor& v, at::Tensor& out,
                                     const at::Tensor& b_loc,
                                     const at::Tensor& b_start_loc,
                                     const at::Tensor& b_seq_len,
                                     int max_input_len, int other_kv_index) {
  callDiopi(diopiTokenSoftmaxReduceVInference, out, logics, v, b_loc,
            b_start_loc, b_seq_len, max_input_len, other_kv_index);
}

// void extTokenDecodeAttentionInference(const at::Tensor& q, const at::Tensor& k,
//                                       const at::Tensor& v, at::Tensor& out,
//                                       const at::Tensor& b_loc,
//                                       const at::Tensor& b_start_loc,
//                                       const at::Tensor& b_seq_len,
//                                       int max_input_len, int other_kv_index) {
//   callDiopi(diopiTokenDecodeAttentionInference, out, q, k, v, b_loc, b_start_loc,
//             b_seq_len, max_input_len, other_kv_index);
// }

// void extTokenDecodeAttentionInferenceBatchOne(const at::Tensor& q, const at::Tensor& k,
//                                       const at::Tensor& v, at::Tensor& out,
//                                       const at::Tensor& b_loc,
//                                       const at::Tensor& b_start_loc,
//                                       const at::Tensor& b_seq_len,
//                                       int max_input_len, int other_kv_index) {
//   callDiopi(diopiTokenDecodeAttentionInferenceBatchOne, out, q, k, v, b_loc, b_start_loc,
//             b_seq_len, max_input_len, other_kv_index);
// }

// void extIncreFlashAttention(const at::Tensor& q, const at::Tensor& k,
//                             const at::Tensor& v, at::Tensor& out,
//                             const int head, const char* layout,
//                             const c10::optional<at::Tensor>& padding_mask = {},
//                             const c10::optional<at::Tensor>& atten_mask = {},
//                             const OptionalIntArray& actual_seq_lengths = {},
//                             int64_t num_heads = 1, double scale_value = 1.0,
//                             const std::string& input_layout = "BSH", int64_t num_key_value_heads = 0) {
//   callDiopi(diopiIncreFlashAttention, out, q, k, v, padding_mask, atten_mask,
//             actual_seq_lengths, num_heads, scale_value, input_layout.c_str(), num_key_value_heads);
// }

void extPromptFlashAttention(at::Tensor& out, const at::Tensor& q,
                             const at::Tensor& k, const at::Tensor& v,
                             const c10::optional<at::Tensor>& padding_mask = {},
                             const c10::optional<at::Tensor>& atten_mask = {},
                             const at::IntArrayRef& actual_seq_lengths = {},
                             int64_t num_heads = 1, double scale_value = 1.0,
                             int64_t pre_tokens = 2147473647,
                             int64_t next_tokens = 0,
                             const std::string& input_layout = "BSH",
                             int64_t num_key_value_heads = 0) {
  callDiopi(diopiPromptFlashAttention, out, q, k, v, padding_mask, atten_mask,
            actual_seq_lengths, num_heads, scale_value, pre_tokens,
            next_tokens, input_layout.c_str(), num_key_value_heads);
}

void extContextAttentionInference(const at::Tensor& q, const at::Tensor& k,
                                  const at::Tensor& v, at::Tensor& out,
                                  const at::Tensor& b_start_loc,
                                  const at::Tensor& b_seq_len,
                                  int max_input_len) {
  callDiopi(diopiContextAttentionInference, out, q, k, v, b_start_loc,
            b_seq_len, max_input_len);
}

void extApplyPenalty(at::Tensor& logits, const at::Tensor& presence_penalty,
                     const at::Tensor& frequency_penalty,
                     const at::Tensor& p_token_ids,
                     const at::Tensor& p_token_counts,
                     const at::Tensor& p_cumsum_seq_len,
                     int p_max_len_in_batch) {
  callDiopi(diopiApplyPenalty, logits, presence_penalty, frequency_penalty,
            p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch);
}

void extApplyPenaltyV2(at::Tensor& logits, const at::Tensor& presence_penalty,
                     const at::Tensor& frequency_penalty,
                     const at::Tensor& repetition_penalty,
                     const at::Tensor& p_token_ids,
                     const at::Tensor& p_token_counts,
                     const at::Tensor& p_cumsum_seq_len,
                     int p_max_len_in_batch) {
  callDiopi(diopiApplyPenaltyV2, logits, presence_penalty, frequency_penalty, repetition_penalty,
            p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch);
}

void extPagedAttention(at::Tensor& out, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, 
                      const c10::optional<at::Tensor>& padding_mask = {},
                      const c10::optional<at::Tensor>& atten_mask = {},
                      const at::IntArrayRef& actual_seq_lengths = {},
                      const c10::optional<at::Tensor>& block_table = {},
                      int64_t num_heads = 1, int64_t num_key_value_heads = 0,
                      double scale_value = 1.0, const std::string& input_layout = "BSH", 
                      int64_t block_size = 0, int64_t inner_precise = 1,
                      const c10::optional<at::Tensor>& antiquant_scale = {}, const c10::optional<at::Tensor>& antiquant_offset = {},
                      const c10::optional<at::Tensor>& dequant_scale1 = {}, const c10::optional<at::Tensor>& quant_scale1 = {},
                      const c10::optional<at::Tensor>& dequant_scale2 = {}, const c10::optional<at::Tensor>& quant_scale2 = {},
                      const c10::optional<at::Tensor>& quant_offset2 = {}, const c10::optional<at::Tensor>& kv_padding_size = {}
                      ) {
  callDiopi(diopiPagedAttention, out, q, k, v, padding_mask, atten_mask, actual_seq_lengths,
            antiquant_scale, antiquant_offset, 
            block_table,
            dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, kv_padding_size,
            num_heads, scale_value, input_layout.c_str(), num_key_value_heads, block_size, inner_precise
            );
}

void extRotaryEmbeddingV2(at::Tensor& query, at::Tensor& key, const at::Tensor& cos, const at::Tensor& sin) {
  callDiopi(diopiRotaryEmbeddingV2, query, key, cos, sin);
}

void extMatmulAllReduce(at::Tensor& out, const at::Tensor& x1,
                        const at::Tensor& x2, const c10::optional<at::Tensor>& bias,
                        const char* group, const char* reduce_op,
                        int64_t comm_turn, int64_t stream_mode) {
  callDiopi(diopiMatmulAllReduce, out, x1, x2,
            bias, group, reduce_op, comm_turn, stream_mode);
}

// 判断是否有对应的 diopi 实现:
//   如果有, 则直接 pybind 上去;
//   否则不注册, 等到 python 层处理.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Check if weak symbol defined
  if (&diopiAdamW != nullptr) {
    m.def("adamw", &extAdamW, "deeplink ext_adamw");
  }
  if (&diopiFlashAttention != nullptr) {
    m.def("fa_fwd", &extFlashAttention, "deeplink ext_fa_fwd");
  }
  if (&diopiFlashAttentionBackward != nullptr) {
    m.def("fa_bwd", &extFlashAttentionBackward, "deeplink ext_fa_bwd");
  }
  if (&diopiFlashAttentionVarLen != nullptr) {
    m.def("fa_varlen_fwd", &extFlashAttentionVarLen,
          "deeplink ext_fa_varlen_fwd");
  }
  if (&diopiFlashAttentionVarLenBackward != nullptr) {
    m.def("fa_varlen_bwd", &extFlashAttentionVarLenBackward,
          "deeplink ext_fa_varlen_bwd");
  }
  if (&diopiCustomizedFlashAttention != nullptr) {
    m.def("custom_fa_fwd", &extCustomizedFlashAttention,
          "deeplink ext_custom_fa_fwd");
  }
  if (&diopiCustomizedFlashAttentionBackward != nullptr) {
    m.def("custom_fa_bwd", &extCustomizedFlashAttentionBackward,
          "deeplink ext_custom_fa_bwd");
  }
  if (&diopiCustomizedFlashAttentionVarLen != nullptr) {
    m.def("custom_fa_varlen_fwd", &extCustomizedFlashAttentionVarLen,
          "deeplink ext_custom_fa_varlen_fwd");
  }
  if (&diopiCustomizedFlashAttentionVarLenBackward != nullptr) {
    m.def("custom_fa_varlen_bwd", &extCustomizedFlashAttentionVarLenBackward,
          "deeplink ext_custom_fa_varlen_bwd");
  }
  if (&diopiRMSNorm != nullptr) {
    m.def("rms_norm", &extRmsNorm, "deeplink ext_rms_norm");
  }
  if (&diopiRMSNormBackward != nullptr) {
    m.def("rms_norm_backward", &extRmsNormBackward,
          "deeplink ext_rms_norm_backward");
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
    m.def("mha_varlen_fwd", &extMultiHeadAttentionVarLen,
          "deeplink ext_mha_varlen_fwd");
  }
  if (&diopiMultiHeadAttentionVarLenBackward != nullptr) {
    m.def("mha_varlen_bwd", &extMultiHeadAttentionVarLenBackward,
          "deeplink ext_mha_varlen_bwd");
  }
  if (&diopiDestIndexCopyKV != nullptr) {
    m.def("dest_index_copy_kv", &extDestIndexCopyKV,
          "deeplink ext_dest_index_copy_kv");
  }
  if (&diopiTokenAttentionInference != nullptr) {
    m.def("token_attention_inference", &extTokenAttentionInference,
          "deeplink ext_token_attention_inference");
  }
  if (&diopiTokenSoftmaxReduceVInference != nullptr) {
    m.def("token_softmax_reducev_inference", &extTokenSoftmaxReduceVInference,
          "deeplink ext_token_softmax_reducev_inference");
  }
  // if (&diopiTokenDecodeAttentionInference != nullptr) {
  //   m.def("token_decode_attention_inference", &extTokenDecodeAttentionInference,
  //         "deeplink token_decode_attention_inference");
  // }
  // if (&diopiTokenDecodeAttentionInferenceBatchOne != nullptr) {
  //   m.def("token_decode_attention_inference_batch_one", &extTokenDecodeAttentionInferenceBatchOne,
  //         "deeplink token_decode_attention_inference");
  // }
  // if (&diopiIncreFlashAttention != nullptr) {
  //   m.def("incre_flash_attention", &extIncreFlashAttention,
  //         "deeplink incre_flash_attention");
  // }
  if (&diopiPromptFlashAttention != nullptr) {
    m.def("prompt_flash_attention", &extPromptFlashAttention,
          "deeplink ext_prompt_flash_attention");
  }
  if (&diopiContextAttentionInference != nullptr) {
    m.def("context_attention_inference", &extContextAttentionInference,
          "deeplink ext_context_attention_inference");
  }
  if (&diopiApplyPenalty != nullptr) {
    m.def("apply_penalty", &extApplyPenalty, "deeplink ext_apply_penalty");
  }
  if (&diopiApplyPenaltyV2 != nullptr) {
    m.def("apply_penalty_v2", &extApplyPenaltyV2, "deeplink ext_apply_penalty");
  }
  if (&diopiScaledMaskedSoftmax != nullptr) {
    m.def("scaled_masked_softmax_fwd", &extScaledMaskedSoftmax,
          "deeplink ext_scaled_masked_softmax_fwd");
  }
  if (&diopiScaledMaskedSoftmaxBackward != nullptr) {
    m.def("scaled_masked_softmax_bwd", &extScaledMaskedSoftmaxBackward,
          "deeplink ext_scaled_masked_softmax_bwd");
  }
  if (&diopiPagedAttention != nullptr) {
    m.def("paged_attention", &extPagedAttention,
          "deeplink ext_paged_attention");
  }
  if (&diopiRotaryEmbeddingV2 != nullptr) {
    m.def("rotary_embedding_v2", &extRotaryEmbeddingV2, "deeplink extRotaryEmbeddingV2");
  }
  if (&diopiMatmulAllReduce != nullptr) {
    m.def("matmul_all_reduce", &extMatmulAllReduce,
          "deeplink ext_matmul_all_reduce",
          py::arg("out"), py::arg("x1"),
          py::arg("x2"), py::arg("bias"),
          py::arg("group"), py::arg("reduce_op") = "sum",
          py::arg("comm_turn") = 0, py::arg("stream_mode") = 1);
  }
}

}  // namespace dipu::dipu_ext
