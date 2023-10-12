#pragma once

// 这个目前存在bug，生成的IntArrayRefToPyObject似乎传过去就不对了，后续再解决

#include <pybind11/pybind11.h>
#include <torch/types.h>
#include <torch/python.h> 
#include <c10/util/OptionalArrayRef.h>

namespace pybind11 {
namespace detail {

namespace py = pybind11;

at::OptionalIntArrayRef PyObjectToOptionalIntArrayRef(PyObject* obj) {
    if (obj == Py_None) {
        return at::OptionalIntArrayRef();
    }

    // Assuming obj is a PyList
    std::vector<int64_t> vec = py::cast<std::vector<int64_t>>(obj);
    return at::OptionalIntArrayRef(vec);
}

PyObject* OptionalIntArrayRefToPyObject(const at::OptionalIntArrayRef& ref) {
    if (!ref.has_value()) {
        Py_RETURN_NONE;
    }
    auto array_ref = ref.value();
    return py::cast(std::vector<int64_t>(array_ref.data(), array_ref.data() + array_ref.size())).release().ptr();
}


template <>
struct type_caster<at::OptionalIntArrayRef> {
public:
    PYBIND11_TYPE_CASTER(at::OptionalIntArrayRef, _("OptionalIntArrayRef"));

    bool load(py::handle src, bool) {
        if (PyList_Check(src.ptr())) {
            std::vector<int64_t> vec = py::cast<std::vector<int64_t>>(src);
            at::ArrayRef<int64_t> array_ref(vec.data(), 2);
            at::OptionalIntArrayRef v = at::OptionalIntArrayRef(array_ref);
            value = v;
            // // std::cout<<vec[0]<<std::endl;
            // std::cout<<array_ref<<std::endl;
            // std::cout<<*v<<std::endl;
            // std::cout<<bool(array_ref==v)<<std::endl;
            return true;
        } else if (src.is_none()) {
            value = at::OptionalIntArrayRef();  // Assigning an empty OptionalArrayRef
            return true;
        }
        return false;
    }

    static py::handle cast(const at::OptionalIntArrayRef& src, py::return_value_policy /* policy */, py::handle /* parent */) {
        return py::handle(OptionalIntArrayRefToPyObject(src));
    }
};

} // namespace detail
} // namespace pybind11
