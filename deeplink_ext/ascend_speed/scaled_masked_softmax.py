# Copyright (c) 2024, DeepLink.

from deeplink_ext.utils import PlatformType, deeplink_ext_get_platform_type

platform_type = deeplink_ext_get_platform_type()
if platform_type == PlatformType.TORCH_NPU:
    from ._scaled_masked_softmax_npu import ScaledMaskedSoftmax
elif platform_type == PlatformType.TORCH_DIPU:
    from ._scaled_masked_softmax_dipu import ScaledMaskedSoftmax
else:
    raise ImportError

__all__ = ["ScaledMaskedSoftmax"]
