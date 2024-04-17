// Copyright (c) 2023, DeepLink.

#ifndef DIOPI_HELPER_H_WTUAZNIC
#define DIOPI_HELPER_H_WTUAZNIC

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/runtime/device/basedef.h>

#include "pybind_type_cast.h"

namespace dipu::dipu_ext {

namespace type_traits {

template <class T>
struct IsOptionalArithmetic : std::false_type {};

template <class T>
struct IsOptionalArithmetic<c10::optional<T>> : std::is_arithmetic<T> {};

}  // namespace type_traits

inline void checkTensorOnDevice(const at::Tensor& tensor) {
  if (tensor.device().type() == at::DeviceType::CPU) {
    DIPU_LOGE("This op only runs on Device");
    throw std::runtime_error("This op only runs on Device");
  }
}

inline void checkTensorOnDevice(const c10::optional<at::Tensor>& tensor) {
  if (tensor) {
    checkTensorOnDevice(*tensor);
  }
}

// at::Tensor                 ->  diopiTensorHandle_t
// const at::Tensor           ->  diopiConstTensorHandle_t
// const c10::optional<at::Tensor>  ->  diopiConstTensorHandle_t
template <class T, class U = std::decay_t<T>,
          std::enable_if_t<std::is_same_v<U, at::Tensor> ||
                               std::is_same_v<std::remove_reference_t<T>,
                                              const c10::optional<at::Tensor>>,
                           int> = 0>
[[nodiscard]] decltype(auto) castToDiopiType(T&& tensor) {
  checkTensorOnDevice(tensor);
  return diopi_helper::toDiopiTensorHandle(std::forward<T>(tensor));
}

::diopiTensorHandle_t toDiopiTensorHandle(
    c10::optional<at::Tensor>& tensor_opt) {
  if (!tensor_opt.has_value()) {
    return nullptr;
  }
  return diopi_helper::toDiopiTensorHandle(tensor_opt.value());
}

// c10::optional<at::Tensor>  ->  diopiTensorHandle_t
template <class T, std::enable_if_t<std::is_same_v<std::remove_reference_t<T>,
                                                   c10::optional<at::Tensor>>,
                                    int> = 0>
[[nodiscard]] decltype(auto) castToDiopiType(T&& tensor) {
  checkTensorOnDevice(tensor);
  return toDiopiTensorHandle(std::forward<T>(tensor));
}

// at::OptionalArrayRef  ->  diopiSize_t
template <class T, class U = std::decay_t<T>,
          std::enable_if_t<std::is_same_v<U, at::OptionalIntArrayRef>, int> = 0>
[[nodiscard]] decltype(auto) castToDiopiType(T&& shape) {
  return diopi_helper::toDiopiSize(std::forward<T>(shape));
}

::diopiSize_t toDiopiSize(const at::IntArrayRef& input) {
  ::diopiSize_t diopi_size{nullptr, 0};
  diopi_size.data = input.data();
  diopi_size.len = static_cast<int64_t>(input.size());
  return diopi_size;
}

// at::IntArrayRef  ->  diopiSize_t
template <class T, class U = std::decay_t<T>,
          std::enable_if_t<std::is_same_v<U, at::IntArrayRef>, int> = 0>
[[nodiscard]] decltype(auto) castToDiopiType(T&& shape) {
  return toDiopiSize(std::forward<T>(shape));
}

// at::Generator                 ->  diopiGeneratorHandle_t
// c10::optional<at::Generator>  ->  diopiGeneratorHandle_t
template <class T, class U = std::decay_t<T>,
          std::enable_if_t<std::is_same<U, at::Generator>() ||
                               std::is_same<U, c10::optional<at::Generator>>(),
                           int> = 0>
[[nodiscard]] decltype(auto) castToDiopiType(T&& gen) {
  return diopi_helper::toDiopiGeneratorHandle(std::forward<T>(gen));
}

// c10::optional<ArithmeticType>  ->  const ArithmeticType*
template <
    class T, class U = std::decay_t<T>,
    std::enable_if_t<type_traits::IsOptionalArithmetic<U>::value, int> = 0>
[[nodiscard]] auto castToDiopiType(T&& opt) -> const typename U::value_type* {
  if (opt) {
    return &(*std::forward<T>(opt));
  }
  return nullptr;
}

// ArithmeticType  ->  ArithmeticType
// Pointer         ->  Pointer
template <
    class T, class U = std::decay_t<T>,
    std::enable_if_t<std::is_arithmetic_v<U> || std::is_pointer_v<U>, int> = 0>
[[nodiscard]] decltype(auto) castToDiopiType(T&& arg) {
  return std::forward<T>(arg);
}

// NOTE: This function will keep the context in the upper stack frame.
//      You usually don't need to explicit use the return value, i.e.
//      `[[maybe_unused]] auto context = callDiopiKeepContext(...);`
//      is what you should do in most cases.
template <class DiopiFunc, class... Args>
[[nodiscard]] diopiContext callDiopiKeepContext(const DiopiFunc& diopi_func,
                                                Args&&... args) {
  static_assert(std::is_function_v<std::remove_reference_t<DiopiFunc>>,
                "DiopiFunc must be a function");
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiError_t err_code =
      diopi_func(&ctx, castToDiopiType(std::forward<Args>(args))...);
  if (err_code != diopiSuccess) {
    throw std::runtime_error("DIOPI error, code: " + std::to_string(err_code) +
                             ", message: " + diopiGetLastErrorString());
  }
  return ctx;
}

// WARNING: This function will destruct the context after the function call. If
//         you need to keep the context (e.g. casting a `diopiTensorHandle_t*`
//         allocated by `diopiRequireTensor` during the `diopi_func` call to a
//         `at::Tensor` and then using it somewhere outside), please use
//         `callDiopiKeepContext` instead.
template <class DiopiFunc, class... Args>
void callDiopi(const DiopiFunc& diopi_func, Args&&... args) {
  [[maybe_unused]] auto context =
      callDiopiKeepContext(diopi_func, std::forward<Args>(args)...);
}

}  // namespace dipu::dipu_ext

#endif /* end of include guard: DIOPI_HELPER_H_WTUAZNIC */
