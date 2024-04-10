import torch
import random

from deeplink_ext.internevo_ops.adamw import fused_adamw
from deeplink_ext.internevo_ops.adamw_fallback import fused_adamw_fallback

params_shapes = [(3, 3), (3,), (4, 5, 6)]
test_dtypes = [torch.float16, torch.float32, torch.float16]
step_list = [0, 10, 300]


def test_adamw():
    atol = 1e-1
    rtol = 1e-1
    for shape, dtype, step in zip(params_shapes, test_dtypes, step_list):
        # define cpu patameters
        low_bound = -100
        high_bound = 100
        param_num = random.randint(1, 10)
        params_cpu = [
            torch.randn(*shape, dtype=dtype) * (high_bound - low_bound) + low_bound
            for i in range(param_num)
        ]
        grads_cpu = [
            torch.randn(*shape, dtype=dtype) * (high_bound - low_bound) + low_bound
            for i in range(param_num)
        ]
        exp_avgs_cpu = [
            torch.randn(*shape, dtype=dtype) * (high_bound - low_bound) + low_bound
            for i in range(param_num)
        ]
        exp_avg_sqs_cpu = [
            torch.randn(*shape, dtype=dtype) * (high_bound - low_bound) + low_bound
            for i in range(param_num)
        ]

        state_steps_cpu = [torch.tensor([step])]

        # define device parameters
        params_cuda = [i.cuda() for i in params_cpu]
        grads_cuda = [i.cuda() for i in grads_cpu]
        exp_avgs_cuda = [i.cuda() for i in exp_avgs_cpu]
        exp_avg_sqs_cuda = [i.cuda() for i in exp_avg_sqs_cpu]
        state_steps_cuda = [i.cuda() for i in state_steps_cpu]

        max_exp_avg_sqs = None

        # define hyperparameters
        amsgrad = False
        maximize = False
        grad_scale = None
        beta1 = random.uniform(0.9, 0.999)
        beta2 = random.uniform(0.9, 0.999)
        lr = random.uniform(0.001, 0.1)
        weight_decay = random.uniform(0.001, 0.1)
        eps = random.uniform(1e-08, 0.1)

        # perform single optimization on CPU with fused_adamw_fallback
        params_ref, exp_avgs_ref, exp_avg_sqs_ref = fused_adamw_fallback(
            params_cpu,
            grads_cpu,
            exp_avgs_cpu,
            exp_avg_sqs_cpu,
            max_exp_avg_sqs,
            state_steps_cpu,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=grad_scale,
            found_inf=None,
        )

        # perform single optimization on device with fused_adamw
        params_ext, exp_avgs_ext, exp_avg_sqs_ext = fused_adamw(
            params_cuda,
            grads_cuda,
            exp_avgs_cuda,
            exp_avg_sqs_cuda,
            max_exp_avg_sqs,
            state_steps_cuda,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=grad_scale,
            found_inf=None,
        )

        for i in range(len(params_ref)):
            assert torch.allclose(
                params_ext[i].cpu(),
                params_ref[i].to(dtype),
                rtol,
                atol,
                equal_nan=True,
            )
            assert torch.allclose(
                exp_avgs_ext[i].cpu(),
                exp_avgs_ref[i].to(dtype),
                rtol,
                atol,
                equal_nan=True,
            )

            assert torch.allclose(
                exp_avg_sqs_ext[i].cpu(),
                exp_avg_sqs_ref[i].to(dtype),
                rtol,
                atol,
                equal_nan=True,
            )
    print("\033[92m#####Test for fused_adamw Succeed!!#####\033[0m")


def test_adamw_bfloat16():
    rtol = 1e-1
    atol = 1e-1
    for shape, step in zip(params_shapes, step_list):
        # define bf16 patameters on device
        low_bound = -100
        high_bound = 100
        param_num = random.randint(1, 10)
        params_bf16 = [
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            * (high_bound - low_bound)
            + low_bound
            for i in range(param_num)
        ]
        grads_bf16 = [
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            * (high_bound - low_bound)
            + low_bound
            for i in range(param_num)
        ]
        exp_avgs_bf16 = [
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            * (high_bound - low_bound)
            + low_bound
            for i in range(param_num)
        ]
        exp_avg_sqs_bf16 = [
            torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            * (high_bound - low_bound)
            + low_bound
            for i in range(param_num)
        ]

        state_steps = [torch.tensor([step]).cuda()]

        max_exp_avg_sqs = None

        # define hyperparameters
        amsgrad = False
        maximize = False
        grad_scale = None
        beta1 = random.uniform(0.9, 0.999)
        beta2 = random.uniform(0.9, 0.999)
        lr = random.uniform(0.001, 0.1)
        weight_decay = random.uniform(0.001, 0.1)
        eps = random.uniform(1e-08, 0.1)

        # perform single optimization on device with fused_adamw_fallback
        params_ref, exp_avgs_ref, exp_avg_sqs_ref = fused_adamw_fallback(
            params_bf16,
            grads_bf16,
            exp_avgs_bf16,
            exp_avg_sqs_bf16,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=grad_scale,
            found_inf=None,
        )

        # perform single optimization on device with fused_adamw
        params_ext, exp_avgs_ext, exp_avg_sqs_ext = fused_adamw(
            params_bf16,
            grads_bf16,
            exp_avgs_bf16,
            exp_avg_sqs_bf16,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=grad_scale,
            found_inf=None,
        )

        for i in range(len(params_ref)):
            assert torch.allclose(
                params_ext[i].cpu(),
                params_ref[i].cpu(),
                rtol,
                atol,
                equal_nan=True,
            )
            assert torch.allclose(
                exp_avgs_ext[i].cpu(),
                exp_avgs_ref[i].cpu(),
                rtol,
                atol,
                equal_nan=True,
            )

            assert torch.allclose(
                exp_avg_sqs_ext[i].cpu(),
                exp_avg_sqs_ref[i].cpu(),
                rtol,
                atol,
                equal_nan=True,
            )
    print("\033[92m#####Test for fused_adamw with bf16 Succeed!!#####\033[0m")


if __name__ == "__main__":
    test_adamw()
    test_adamw_bfloat16()
