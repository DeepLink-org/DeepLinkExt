// Copyright (c) 2023, DeepLink.

#ifndef DIOPI_HELPER_H_WTUAZNIC
#define DIOPI_HELPER_H_WTUAZNIC

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/device/basedef.h>

namespace dipu {
namespace dipu_ext {

inline void checkTensorOnDipu(const at::Tensor& tensor) {
  if (tensor.device().type() != dipu::DIPU_DEVICE_TYPE) {
    DIPU_LOGE("This op only runs on DIPU");
    throw std::runtime_error("This op only runs on DIPU");
  }
}

template <class T,
          std::enable_if_t<!std::is_class<std::decay_t<T>>::value, int> = 0>
decltype(auto) toDiopiType(T&& arg) {
  return std::forward<T>(arg);
}

template <
    class T,
    std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value, int> = 0>
decltype(auto) toDiopiType(T&& tensor) {
  checkTensorOnDipu(tensor);
  return diopi_helper::toDiopiTensorHandle(std::forward<T>(tensor));
}

template <
    class T,
    std::enable_if_t<
        std::is_same<std::decay_t<T>, at::OptionalIntArrayRef>::value, int> = 0>
decltype(auto) toDiopiType(T&& shape) {
  return diopi_helper::toDiopiSize(std::forward<T>(shape));
}

template <class DiopiFunc, class... Args>
void callDiopi(DiopiFunc&& diopi_func, Args&&... args) {
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiError_t err_code =
      diopi_func(&ctx, toDiopiType(std::forward<Args>(args))...);
  if (err_code != diopiSuccess) {
    throw std::runtime_error("DIOPI call failed");
  }
}

}  // namespace dipu_ext
}  // namespace dipu

#endif /* end of include guard: DIOPI_HELPER_H_WTUAZNIC */
