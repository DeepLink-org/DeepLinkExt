// Copyright (c) 2023, DeepLink.

#ifndef PYBIND_TYPE_CAST_H_PXMGELYW
#define PYBIND_TYPE_CAST_H_PXMGELYW

#include <vector>

#include <ATen/core/ATen_fwd.h>
#include <c10/util/Optional.h>

#include <Python.h>
#include <listobject.h>
#include <pybind11/cast.h>
#include <pybind11/detail/descr.h>
#include <pybind11/pytypes.h>

namespace dipu::dipu_ext {

using IntArray = std::vector<at::IntArrayRef::value_type>;
using OptionalIntArray = c10::optional<IntArray>;

}  // namespace dipu::dipu_ext

#endif /* end of include guard: PYBIND_TYPE_CAST_H_PXMGELYW */
