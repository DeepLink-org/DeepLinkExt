// Copyright (c) 2023, DeepLink.

#ifndef DIOPI_HELPER_H_WTUAZNIC
#define DIOPI_HELPER_H_WTUAZNIC

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/device/basedef.h>

namespace dipu {
namespace dipu_ext {

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
// c10::optional<at::Tensor>  ->  diopiConstTensorHandle_t
template <
    class T, class U = std::decay_t<T>,
    std::enable_if_t<std::is_same<U, at::Tensor>::value ||
                         std::is_same<U, c10::optional<at::Tensor>>::value,
                     int> = 0>
decltype(auto) castToDiopiType(T&& tensor) {
  checkTensorOnDevice(tensor);
  return diopi_helper::toDiopiTensorHandle(std::forward<T>(tensor));
}

// at::OptionalArrayRef  ->  diopiSize_t
template <
    class T, class U = std::decay_t<T>,
    std::enable_if_t<std::is_same<U, at::OptionalIntArrayRef>::value, int> = 0>
decltype(auto) castToDiopiType(T&& shape) {
  return diopi_helper::toDiopiSize(std::forward<T>(shape));
}

// at::Generator                 ->  diopiGeneratorHandle_t
// c10::optional<at::Generator>  ->  diopiGeneratorHandle_t
template <class T, class U = std::decay_t<T>,
          std::enable_if_t<std::is_same<U, at::Generator>() ||
                               std::is_same<U, c10::optional<at::Generator>>(),
                           int> = 0>
decltype(auto) castToDiopiType(T&& gen) {
  return diopi_helper::toDiopiGeneratorHandle(std::forward<T>(gen));
}

// c10::optional<ArithmeticType>  ->  const ArithmeticType*
template <
    class T, class U = std::decay_t<T>,
    std::enable_if_t<type_traits::IsOptionalArithmetic<U>::value, int> = 0>
auto castToDiopiType(T&& opt) -> const typename U::value_type* {
  if (opt) {
    return &(*opt);
  }
  return nullptr;
}

// ArithmeticType  ->  ArithmeticType
// Pointer         ->  Pointer
template <
    class T, class U = std::decay_t<T>,
    std::enable_if_t<std::is_arithmetic<U>::value || std::is_pointer<U>::value,
                     int> = 0>
decltype(auto) castToDiopiType(T&& arg) {
  return std::forward<T>(arg);
}

template <class DiopiFunc, class... Args>
void callDiopi(DiopiFunc&& diopi_func, Args&&... args) {
  static_assert(std::is_function<std::remove_reference_t<DiopiFunc>>::value,
                "DiopiFunc must be a function");
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiError_t err_code =
      diopi_func(&ctx, castToDiopiType(std::forward<Args>(args))...);
  if (err_code != diopiSuccess) {
    throw std::runtime_error("DIOPI call failed");
  }
}

}  // namespace dipu_ext
}  // namespace dipu

#endif /* end of include guard: DIOPI_HELPER_H_WTUAZNIC */
