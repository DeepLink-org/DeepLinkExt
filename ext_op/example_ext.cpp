// Copyright (c) 2023, DeepLink.

#include <algorithm>
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

at::IntArrayRef optionalIntArrayToIntArrayRefOrDefault(
    const OptionalIntArray& opt, at::IntArrayRef def) {
  if (opt) {
    return {*opt};
  }
  return def;
}

}  // namespace

auto extRmsNorm(const at::Tensor& input,
                const OptionalIntArray& normalized_shape,
                const at::Tensor& weight, const at::Tensor& bias, double eps) {
  at::OptionalIntArrayRef normalized_shape_at =
      optionalIntArrayToIntArrayRefOrDefault(normalized_shape, weight.sizes());
  auto inv_rms = at::empty_like(input);
  auto output = at::empty_like(input);
  callDiopi(diopiRMSNorm, output, inv_rms, input, normalized_shape_at, weight,
            bias, eps);
  return std::make_tuple(std::move(output), std::move(inv_rms));
}

auto extRmsNormBackward(const at::Tensor& input, const at::Tensor& grad_output,
                        const at::Tensor& inv_rms,
                        const OptionalIntArray& normalized_shape,
                        const at::Tensor& weight, const at::Tensor& bias,
                        double eps) {
  at::OptionalIntArrayRef normalized_shape_at =
      optionalIntArrayToIntArrayRefOrDefault(normalized_shape, weight.sizes());
  auto grad_input = at::empty_like(grad_output);
  auto grad_weight = at::empty_like(weight);
  auto grad_bias = at::empty_like(bias);
  callDiopi(diopiRMSNormBackward, grad_input, grad_weight, grad_bias,
            grad_output, input, weight, bias, inv_rms, normalized_shape_at,
            eps);
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

void extApplyRotary(at::Tensor output, const at::Tensor& input,
                    const at::Tensor& cos, const at::Tensor& sin,
                    const bool conj, const bool interleaved = false) {
  callDiopi(diopiRotaryEmbedding, output, input, cos, sin, conj, interleaved);
}

auto extMultiHeadAttention(const at::Tensor& q, const at::Tensor& k,
                           const at::Tensor& v, double dropout_p,
                           bool is_casual, bool return_debug_mask,
                           double scale) {
  const auto batch_size = q.sizes()[0];
  const auto q_seq_len = q.sizes()[1];
  const auto head_num = q.sizes()[2];
  const auto k_seq_len = k.sizes()[1];

  auto out = at::empty_like(q);

  const IntArray softmax_lse_size{batch_size, head_num, q_seq_len};
  auto softmax_lse = at::empty(softmax_lse_size, q.options());

  auto gen = createDIPUGenerator();

  const IntArray debug_attn_mask_size{batch_size, head_num, q_seq_len,
                                      k_seq_len};
  const auto debug_attn_mask_option = q.options().dtype(at::kBool);
  auto debug_attn_mask = return_debug_mask ? at::empty(debug_attn_mask_size,
                                                       debug_attn_mask_option)
                                           : at::empty({0});

  callDiopi(diopiMultiHeadAttention, q, k, v, dropout_p, is_casual,
            return_debug_mask, scale, out, softmax_lse, gen, debug_attn_mask);
  return std::make_tuple(std::move(out), std::move(softmax_lse), std::move(gen),
                         std::move(debug_attn_mask));
}

auto extMultiHeadAttentionBackward(const at::Tensor& grad_out,
                                   const at::Tensor& q, const at::Tensor& k,
                                   const at::Tensor& v, const at::Tensor& out,
                                   const at::Tensor& softmax_lse,
                                   double dropout_p, bool is_casual,
                                   at::Generator& gen, double scale) {
  auto grad_q = at::empty_like(q);
  auto grad_k = at::empty_like(k);
  auto grad_v = at::empty_like(v);
  callDiopi(diopiMultiHeadAttentionBackward, grad_out, q, k, v, out,
            softmax_lse, dropout_p, is_casual, gen, scale, grad_q, grad_k,
            grad_v);
  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v));
}

auto extMultiHeadAttentionVarLen(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v,
                                 const at::Tensor& cum_seq_q,
                                 const at::Tensor& cum_seq_k,
                                 std::int64_t max_q, std::int64_t max_k,
                                 double dropout_p, bool is_casual,
                                 bool return_debug_mask, double scale) {
  const auto head_num = q.sizes()[1];
  const auto batch_size = cum_seq_q.sizes()[0] - 1;

  auto out = at::empty_like(q);

  const IntArray softmax_lse_size{batch_size, head_num, max_q};
  auto softmax_lse = at::empty(softmax_lse_size, q.options());

  auto gen = createDIPUGenerator();

  const IntArray debug_attn_mask_size{batch_size, head_num, max_q, max_k};
  const auto debug_attn_mask_option = q.options().dtype(at::kBool);
  auto debug_attn_mask = return_debug_mask ? at::empty(debug_attn_mask_size,
                                                       debug_attn_mask_option)
                                           : at::empty({0});

  callDiopi(diopiMultiHeadAttentionVarLen, q, k, v, cum_seq_q, cum_seq_k, max_q,
            max_k, dropout_p, is_casual, return_debug_mask, scale, out,
            softmax_lse, gen, debug_attn_mask);
  return std::make_tuple(std::move(out), std::move(softmax_lse), std::move(gen),
                         std::move(debug_attn_mask));
}

auto extMultiHeadAttentionVarLenBackward(
    const at::Tensor& grad_out, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& out, const at::Tensor& softmax_lse,
    const at::Tensor& cum_seq_q, const at::Tensor& cum_seq_k,
    std::int64_t max_q, std::int64_t max_k, double dropout_p, bool is_casual,
    at::Generator& gen, double scale) {
  auto grad_q = at::empty_like(q);
  auto grad_k = at::empty_like(k);
  auto grad_v = at::empty_like(v);
  callDiopi(diopiMultiHeadAttentionVarLenBackward, grad_out, q, k, v, out,
            softmax_lse, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
            is_casual, gen, scale, grad_q, grad_k, grad_v);
  return std::make_tuple(std::move(grad_q), std::move(grad_k),
                         std::move(grad_v));
}

// 判断是否有对应的 diopi 实现:
//   如果有, 则直接 pybind 上去;
//   否则不注册, 等到 python 层处理.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  if (&diopiRMSNorm != nullptr) {  // Check if weak symbol defined
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
}

}  // namespace dipu_ext
}  // namespace dipu
