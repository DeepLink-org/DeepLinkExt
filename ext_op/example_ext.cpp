#include <torch/extension.h>

#include <iostream>

#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/base/basedef.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;

void ext_apply_rotary(torch::Tensor output, const torch::Tensor input, const torch::Tensor cos, const torch::Tensor sin, const bool conj)
{
    auto output_p = toDiopiTensorHandle(output);
    auto cos_p = toDiopiTensorHandle(cos);
    auto sin_p = toDiopiTensorHandle(sin);
    auto input_p = toDiopiTensorHandle(input);
    diopiDevice_t device;
    diopiGetTensorDevice(output_p, &device);
    diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
    diopiContextHandle_t ch = &ctx;

    if (device == diopi_host || input.device().type() != dipu::DIPU_DEVICE_TYPE)
    {
        std::cout << "We only can run this op on dipu!" << std::endl;
        throw "We only can run this op on dipu!";
    }

    auto ret =
        diopiRotaryEmbedding(ch, output_p, input_p, cos_p, sin_p, conj); // 此处更换为diopi内相应的函数
    if (ret == diopiSuccess)
    {
        // auto tensorhandle = reinterpret_cast<torch::Tensor*>(*outhandle);
        return;
    }
    throw "diopicalculate failed!";
    return;
}

// 判断是否有对应的diopi实现，如果有，则直接pybind上去。如果没有，则不注册，再到python层处理。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // if(reinterpret_cast<void*>(diopiNmsMmcv) != nullptr)
    //     m.def("m_nms", &nms_diopi, "deeplink nms");
    if (reinterpret_cast<void *>(diopiRotaryEmbedding) != nullptr)
        m.def("apply_rotary", &ext_apply_rotary, "deeplink ext_apply_rotary");
}
