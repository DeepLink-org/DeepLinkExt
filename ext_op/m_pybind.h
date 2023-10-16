#pragma once

#include <vector>

#include "pybind11/pybind11.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "torch/types.h"

namespace dipu_ext {

using IntArray = std::vector<at::IntArrayRef::value_type>;
using OptionalIntArray = c10::optional<IntArray>;

} // namespace dipu_ext

namespace pybind11 {
namespace detail {

namespace py = pybind11;

template <>
struct type_caster<at::OptionalIntArrayRef> {
public:
    PYBIND11_TYPE_CASTER(dipu_ext::OptionalIntArray, _("OptionalIntArray"));

    bool load(py::handle src, bool) {
        if (PyList_Check(src.ptr())) {
            value = py::cast<dipu_ext::IntArray>(src);
            return true;
        } else if (src.is_none()) {
            value = c10::nullopt;
            return true;
        }
        return false;
    }
};

} // namespace detail
} // namespace pybind11
