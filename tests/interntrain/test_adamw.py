# Copyright (c) 2024, DeepLink.

import copy
import torch
from torch import nn
from deeplink_ext.interntrain_ops.adamw import AdamW


def test_AdamW():
    class MlpModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 512)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    dtype = torch.float32
    device = "cuda"
    input_data_cpu = torch.rand(16, 128, dtype=dtype)
    input_data_device = input_data_cpu.to(device)
    cpu_model = MlpModel().to(dtype)
    device_model = copy.deepcopy(cpu_model).to(device)

    adamW_cpu = torch.optim.AdamW(
        params=cpu_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=True,
    )

    adamW_ext = AdamW(
        params=device_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=True,
    )

    steps = 15
    for step in range(steps):
        adamW_cpu.zero_grad()
        adamW_ext.zero_grad()

        output_cpu = cpu_model(input_data_cpu)
        output_device = device_model(input_data_device)

        output_cpu.mean().backward()
        output_device.mean().backward()

        adamW_cpu.step()
        adamW_ext.step()

    params_zip = zip(list(cpu_model.parameters()), list(device_model.parameters()))
    for cpu_param, device_param in params_zip:
        assert torch.allclose(cpu_param, device_param.cpu(), rtol=1e-4, atol=1e-4)
