// Copyright (c) 2023, DeepLink.

#include <algorithm>
#include <tuple>
#include <vector>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty_like.h>
#include <c10/util/OptionalArrayRef.h>
#include <torch/csrc/utils/pybind.h>  // IWYU pragma: keep

#include <pybind11/pybind11.h>

#include <diopi/functions_ext.h>

#include "diopi_helper.h"
#include "pybind_type_cast.h"

namespace dipu {
namespace dipu_ext {

std::tuple<at::Tensor, at::Tensor> extRmsNorm(const at::Tensor& input,
                                              OptionalIntArray normalized_shape,
                                              const at::Tensor& weight,
                                              const at::Tensor& bias,
                                              double eps) {
  at::OptionalIntArrayRef normalized_shape_at;
  if (normalized_shape) {
    normalized_shape_at = *normalized_shape;
  } else {
    normalized_shape_at = weight.sizes();
  }
  auto inv_rms = at::empty_like(input);
  auto output = at::empty_like(input);
  callDiopi(diopiRMSNorm, output, inv_rms, input, normalized_shape_at, weight,
            bias, eps);
  return {output, inv_rms};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> extRmsNormBackward(
    const at::Tensor& input, const at::Tensor& grad_output,
    const at::Tensor& inv_rms, OptionalIntArray normalized_shape,
    const at::Tensor& weight, const at::Tensor& bias, double eps) {
  at::OptionalIntArrayRef normalized_shape_at;
  if (normalized_shape) {
    normalized_shape_at = *normalized_shape;
  } else {
    normalized_shape_at = weight.sizes();
  }
  auto grad_input = at::empty_like(grad_output);
  auto grad_weight = at::empty_like(weight);
  auto grad_bias = at::empty_like(bias);
  callDiopi(diopiRMSNormBackward, grad_input, grad_weight, grad_bias,
            grad_output, input, weight, bias, inv_rms, normalized_shape_at,
            eps);
  return {grad_input, grad_weight, grad_bias};
}

void extApplyRotary(at::Tensor output, const at::Tensor& input,
                    const at::Tensor& cos, const at::Tensor& sin,
                    const bool conj) {
  callDiopi(diopiRotaryEmbedding, output, input, cos, sin, conj);
}

at::Tensor extApplyPenalty(
    at::Tensor& Logits, const at::Tensor& presence_penalty, const at::Tensor& frequency_penalty, const at::Tensor& p_token_ids,
    const at::Tensor& p_token_counts, const at::Tensor& p_cumsum_seq_len, int p_max_len_in_batch) {
      callDiopi(diopiApplyPenalty, Logits, presence_penalty, frequency_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch);
      return Logits;
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
  if (&diopiApplyPenalty != nullptr) {
    m.def("apply_penalty", &extApplyPenalty, "deeplink ext_apply_penalty");
  }
}
}  // namespace dipu_ext
}  // namespace dipu
