#include <torch/extension.h>

#include <iostream>

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/base/basedef.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;

torch::Tensor nms_diopi(torch::Tensor boxes, torch::Tensor scores, float iou_threshold, int offset) {
    auto boxes_p = toDiopiTensorHandle(boxes);
    diopiDevice_t device;
    diopiGetTensorDevice(boxes_p, &device);
    diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
    diopiContextHandle_t ch = &ctx;
    torch::Tensor out;
    auto outp = toDiopiTensorHandle(out);
    diopiTensorHandle_t* outhandle = &outp;
    auto scores_p = toDiopiTensorHandle(scores);

    if (device == diopi_host || boxes.device().type() != dipu::DIPU_DEVICE_TYPE) {
        std::cout<<"We only can run this op on dipu!"<<std::endl;
        throw "We only can run this op on dipu!";
    }

    auto ret =
        diopiNmsMmcv(ch, outhandle, boxes_p, scores_p, iou_threshold, offset);  // 此处更换为diopi内相应的函数
    if (ret == diopiSuccess) {
        auto tensorhandle = reinterpret_cast<torch::Tensor*>(*outhandle);
        return *tensorhandle;
    }
    throw "diopicalculate failed!";
    return torch::Tensor();
}

// 判断是否有对应的diopi实现，如果有，则直接pybind上去。如果没有，则不注册，再到python层处理。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    if(reinterpret_cast<void*>(diopiNmsMmcv) != nullptr)
        m.def("m_nms", &nms_diopi, "deeplink nms");
}
