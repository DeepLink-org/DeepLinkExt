import os
import re
import enum

class PlatformType(enum.Enum):
    UNSUPPORTED = 0
    TORCH = 1
    TORCH_DIPU = 2
    TORCH_NPU = 3
    TORCH_CUDA = 4

def deeplink_ext_get_platform_type():
    platform_type = os.environ.get('DEEPLINK_EXT_PLATFORM_TYPE')
    if platform_type is None:
        print(f'[deeplink_ext] Now using torch_dipu as default backend...')
        return PlatformType.TORCH_DIPU
    else:
        if platform_type.upper() == "TORCH_DIPU":
            return PlatformType.TORCH_DIPU
        elif platform_type.upper() == "TORCH_NPU":
            return PlatformType.TORCH_NPU
        else:
            return PlatformType.UNSUPPORTED
