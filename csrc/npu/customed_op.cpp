#include <torch/types.h>
#include "../extension_helper.h"

void customed_op_impl(at::Tensor& a, at::Tensor& b);
// 4. kernel of customed_op
void customed_op_npu(at::Tensor& a, at::Tensor& b) {
    std::printf("== call customed_op kernel\n");
}

REGISTER_DEVICE_IMPL(customed_op_impl, XPU, customed_op_npu);
