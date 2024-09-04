import pytest
import torch

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

@pytest.fixture(scope='session', autouse=True)
def import_module():
    platform = deeplink_ext_get_platform_type()
    if platform == PlatformType.TORCH_NPU:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
    elif platform == PlatformType.TORCH_DIPU:
        import torch_dipu
    else:
        raise ValueError("backend platform does not supported by deeplink_ext")
