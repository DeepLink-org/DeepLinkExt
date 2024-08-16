# Copyright (c) 2024, DeepLink.
from typing import List
import torch

__all__ = ["AdamW"]

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    import torch_npu
    from torch_npu.optim.npu_fused_adamw import NpuFusedAdamW as AdamW
elif platform_type == PlatformType.TORCH_DIPU:
    # import torch_dipu
    # assert torch_dipu.vendor_type == 'NPU', "ascend_speed framework only support NPU accelerators."
    from ._adamw_dipu import AdamW
else:
    raise ImportError
