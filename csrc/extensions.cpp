// Copyright (c) 2023, DeepLink.

#include <cstdint>
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

#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

#include "diopi_helper.h"
#include "pybind_type_cast.h"

namespace dipu::dipu_ext {

void extAdamW(at::Tensor& param, at::Tensor& exp_avg, at::Tensor& exp_avg_sq,
              at::Tensor& max_exp_avg_sq, at::Tensor& grad, float lr,
              float beta1, float beta2, float epsilon, float weight_decay,
              int64_t step, bool amsgrad) {
  // the diopiAdamW func has no "maximize" param
  callDiopi(diopiAdamW, param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr,
            beta1, beta2, epsilon, weight_decay, step, amsgrad);
}

auto extRmsNorm(at::Tensor& output, at::Tensor& inv_rms,
                const at::Tensor& input,
                const OptionalIntArray& normalized_shape,
                const at::Tensor& weight, const at::Tensor& bias, double eps) {
  at::OptionalIntArrayRef normalized_shape_at = *normalized_shape;
  callDiopi(diopiRMSNorm, output, inv_rms, input, normalized_shape_at, weight,
            bias, eps);
  return std::make_tuple(std::move(output), std::move(inv_rms));
}

auto extRmsNormBackward(at::Tensor& grad_input, at::Tensor& grad_weight,
                        at::Tensor& grad_bias, const at::Tensor& grad_output,
                        const at::Tensor& input, const at::Tensor& weight,
                        const at::Tensor& bias, const at::Tensor& inv_rms,
                        const OptionalIntArray& normalized_shape, double eps) {
  at::OptionalIntArrayRef normalized_shape_at = *normalized_shape;
  callDiopi(diopiRMSNormBackward, grad_input, grad_weight, grad_bias,
            grad_output, input, weight, bias, inv_rms, normalized_shape_at,
            eps);
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
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

auto extFlashAttention(at::Tensor& out, const at::Tensor& q,
                       const at::Tensor& k, const at::Tensor& v,
                       double p_dropout, double softmax_scale, bool is_causal,
                       int64_t head_num, c10::string_view input_layout) {
  diopiTensorHandle_t attention_mask = nullptr;
  diopiTensorHandle_t dropout_mask = nullptr;
  diopiTensorHandle_t softmax_max = nullptr;
  diopiTensorHandle_t softmax_sum = nullptr;
  diopiTensorHandle_t softmax_out = nullptr;

  auto gen = createDIPUGenerator();

  [[maybe_unused]] auto context = callDiopiKeepContext(
      diopiFlashAttention, out, &attention_mask, &dropout_mask, &softmax_max,
      &softmax_sum, &softmax_out, gen, q, k, v, p_dropout, softmax_scale,
      is_causal, head_num, input_layout.data());

  return std::make_tuple(
      attention_mask
          ? *dipu::diopi_helper::fromDiopiTensorHandle(attention_mask)
          : at::Tensor(),
      dropout_mask ? *dipu::diopi_helper::fromDiopiTensorHandle(dropout_mask)
                   : at::Tensor(),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_max),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_sum),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_out));
}

auto extFlashAttentionV2(at::Tensor& out, const at::Tensor& q,
                         const at::Tensor& k, const at::Tensor& v,
                         const at::Tensor& attention_mask, double p_dropout,
                         double softmax_scale, int64_t head_num,
                         c10::string_view input_layout) {
  diopiTensorHandle_t dropout_mask = nullptr;
  diopiTensorHandle_t softmax_max = nullptr;
  diopiTensorHandle_t softmax_sum = nullptr;
  diopiTensorHandle_t softmax_out = nullptr;

  auto gen = createDIPUGenerator();

  [[maybe_unused]] auto context = callDiopiKeepContext(
      diopiFlashAttentionV2, out, &dropout_mask, &softmax_max, &softmax_sum,
      &softmax_out, gen, q, k, v, attention_mask, p_dropout, softmax_scale,
      head_num, input_layout.data());

  return std::make_tuple(
      dropout_mask ? *dipu::diopi_helper::fromDiopiTensorHandle(dropout_mask)
                   : at::Tensor(),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_max),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_sum),
      *dipu::diopi_helper::fromDiopiTensorHandle(softmax_out));
}

auto extFlashAttentionBackward(
    at::Tensor& grad_q, at::Tensor& grad_k, at::Tensor& grad_v,
    const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& out,
    const at::Tensor& attention_mask, const at::Tensor& dropout_mask,
    const at::Tensor& softmax_max, const at::Tensor& softmax_sum,
    const at::Tensor& softmax_out, double p_dropout, double softmax_scale,
    int64_t head_num, c10::string_view input_layout) {
  callDiopi(diopiFlashAttentionBackward, grad_q, grad_k, grad_v, grad_out, q, k,
            v, out, attention_mask, dropout_mask, softmax_max, softmax_sum,
            softmax_out, p_dropout, softmax_scale, head_num,
            input_layout.data());
  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v));
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

// For lightllm, rms_norm reuses the diopi implementation of internlm
auto extRmsNormLightllm(const at::Tensor& x, const at::Tensor& weight,
                        float eps) {
  at::ScalarType acc_type = x.scalar_type();
  if (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) {
    acc_type = at::kFloat;
  }
  auto inv_rms = at::empty_like(x, acc_type);
  auto out = at::empty_like(x);
  auto bias = at::empty_like(weight);
  at::OptionalIntArrayRef normalized_shape = weight.sizes();
  callDiopi(diopiRMSNorm, out, inv_rms, x, normalized_shape, weight, bias, eps);
  return out;
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
  if (&diopiFlashAttentionV2 != nullptr) {
    m.def("fa_fwd_v2", &extFlashAttentionV2, "deeplink ext_fa_fwd_v2");
  }
  if (&diopiFlashAttentionBackward != nullptr) {
    m.def("fa_bwd", &extFlashAttentionBackward, "deeplink ext_fa_bwd");
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
  if (&diopiContextAttentionInference != nullptr) {
    m.def("context_attention_inference", &extContextAttentionInference,
          "deeplink ext_context_attention_inference");
  }
  if (&diopiApplyPenalty != nullptr) {
    m.def("apply_penalty", &extApplyPenalty, "deeplink ext_apply_penalty");
  }
}

}  // namespace dipu::dipu_ext
