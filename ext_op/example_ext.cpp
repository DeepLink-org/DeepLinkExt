#include <torch/extension.h>
#include <c10/util/OptionalArrayRef.h>

#include <iostream>

#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/base/basedef.h"
#include "m_pybind.h"



using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiSize;



std::tuple<at::Tensor, at::Tensor> ext_rms_norm(const torch::Tensor input, at::OptionalIntArrayRef normalized_shape, 
                    const torch::Tensor weight, const torch::Tensor bias, double eps) {
    normalized_shape = weight.sizes();
    auto inv_rms = at::empty_like(input);
    auto output = at::empty_like(input);
    
    auto input_p = toDiopiTensorHandle(input);
    auto output_p = toDiopiTensorHandle(output);
    auto inv_rms_p = toDiopiTensorHandle(inv_rms);
    auto bias_p = toDiopiTensorHandle(bias);
    auto weight_p = toDiopiTensorHandle(weight);
    auto normalized_shape_p = toDiopiSize(normalized_shape);


    diopiDevice_t device;
    diopiGetTensorDevice(input_p, &device);
    diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
    diopiContextHandle_t ch = &ctx;

    if (device == diopi_host || input.device().type() != dipu::DIPU_DEVICE_TYPE)
    {
        std::cout << "We only can run this op on dipu!" << std::endl;
        throw "We only can run this op on dipu!";
    }

    // std::cout<<input.device().type()<<std::endl;
    // std::cout<<dipu::DIPU_DEVICE_TYPE<<std::endl;

    auto ret =
        diopiRMSNorm(ch, output_p, inv_rms_p, input_p, normalized_shape_p, weight_p, bias_p, eps);  // 此处更换为diopi内相应的函数
    if (ret == diopiSuccess) {
        return std::tie(output, inv_rms);
    }
    throw "diopicalculate failed!";
    return std::tie(output, inv_rms);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ext_rms_norm_backward(const torch::Tensor& input, const torch::Tensor& grad_output,
                     const torch::Tensor& inv_rms, at::OptionalIntArrayRef normalized_shape, const torch::Tensor& weight, 
                     const torch::Tensor& bias, double eps = 1e-6) {
    
    normalized_shape = weight.sizes();
    // std::cout<<*normalized_shape<<std::endl;

    auto grad_input = at::empty_like(grad_output);
    auto grad_weight = at::empty_like(weight);
    auto grad_bias = at::empty_like(bias);
    
    auto input_p = toDiopiTensorHandle(input);
    auto grad_output_p = toDiopiTensorHandle(grad_output);
    auto inv_rms_p = toDiopiTensorHandle(inv_rms);
    auto bias_p = toDiopiTensorHandle(bias);
    auto weight_p = toDiopiTensorHandle(weight);
    auto normalized_shape_p = toDiopiSize(normalized_shape);
    auto grad_input_p = toDiopiTensorHandle(grad_input);
    auto grad_weight_p = toDiopiTensorHandle(grad_weight);
    auto grad_bias_p = toDiopiTensorHandle(grad_bias);

    diopiDevice_t device;
    diopiGetTensorDevice(input_p, &device);
    diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
    diopiContextHandle_t ch = &ctx;

    if (device == diopi_host || input.device().type() != dipu::DIPU_DEVICE_TYPE)
    {
        std::cout << "We only can run this op on dipu!" << std::endl;
        throw "We only can run this op on dipu!";
    }

    auto ret =
        diopiRMSNormBackward(ch, grad_input_p, grad_weight_p, grad_bias_p, grad_output_p, input_p, weight_p, bias_p, inv_rms_p, normalized_shape_p, eps);  // 此处更换为diopi内相应的函数
    if (ret == diopiSuccess) {
        return std::tie(grad_input, grad_weight, grad_bias);
    }
    throw "diopicalculate failed!";
    return std::tie(grad_input, grad_weight, grad_bias);
}


void ext_apply_rotary(const torch::Tensor x1, const torch::Tensor x2, const torch::Tensor cos, const torch::Tensor sin,
                      torch::Tensor out1, torch::Tensor out2, const bool conj)
{
    auto out1_p = toDiopiTensorHandle(out1);
    auto out2_p = toDiopiTensorHandle(out2);
    auto cos_p = toDiopiTensorHandle(cos);
    auto sin_p = toDiopiTensorHandle(sin);
    auto x1_p = toDiopiTensorHandle(x1);
    auto x2_p = toDiopiTensorHandle(x2);

    diopiDevice_t device;
    diopiGetTensorDevice(input_p, &device);
    diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
    diopiContextHandle_t ch = &ctx;

    if (device == diopi_host || input.device().type() != dipu::DIPU_DEVICE_TYPE)
    {
        std::cout << "We only can run this op on dipu!" << std::endl;
        throw "We only can run this op on dipu!";
    }

    // std::cout<<input.device().type()<<std::endl;
    // std::cout<<dipu::DIPU_DEVICE_TYPE<<std::endl;

    auto ret =
        diopiRMSNorm(ch, output_p, inv_rms_p, input_p, normalized_shape_p, weight_p, bias_p, eps);  // 此处更换为diopi内相应的函数
    if (ret == diopiSuccess) {
        return std::tie(output, inv_rms);
    }
    throw "diopicalculate failed!";
    return std::tie(output, inv_rms);
}

// 判断是否有对应的diopi实现，如果有，则直接pybind上去。如果没有，则不注册，再到python层处理。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    if(reinterpret_cast<void*>(diopiRMSNorm) != nullptr)
        m.def("rms_norm", &ext_rms_norm, "deeplink ext_rms_norm");
    if(reinterpret_cast<void*>(diopiRMSNormBackward) != nullptr)
        m.def("rms_norm_backward", &ext_rms_norm_backward, "deeplink ext_rms_norm_backward");
    if(reinterpret_cast<void*>(diopiRotaryEmbedding) != nullptr)
        m.def("apply_rotary", &ext_apply_rotary, "deeplink ext_apply_rotary");
}
