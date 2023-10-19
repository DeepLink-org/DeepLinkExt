// Copyright (c) 2023, DeepLink.

#ifndef PYBIND_TYPE_CAST_H_PXMGELYW
#define PYBIND_TYPE_CAST_H_PXMGELYW

#include <vector>

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <torch/types.h>

#include <pybind11/pybind11.h>

namespace dipu {
namespace dipu_ext {

using IntArray = std::vector<at::IntArrayRef::value_type>;
using OptionalIntArray = c10::optional<IntArray>;

}  // namespace dipu_ext
}  // namespace dipu

namespace pybind11 {
namespace detail {

namespace py = pybind11;

template <>
struct type_caster<at::OptionalIntArrayRef> {
 public:
  PYBIND11_TYPE_CASTER(dipu::dipu_ext::OptionalIntArray, _("OptionalIntArray"));

  bool load(py::handle src, bool /*unused*/) {
    if (PyList_Check(src.ptr())) {
      value = py::cast<dipu::dipu_ext::IntArray>(src);
      return true;
    }
    if (src.is_none()) {
      value = c10::nullopt;
      return true;
    }
    return false;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif /* end of include guard: PYBIND_TYPE_CAST_H_PXMGELYW */
