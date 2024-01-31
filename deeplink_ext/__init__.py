# Copyright (c) 2024, DeepLink.


def _init():
    # deeplink_ext is developed based on dipu
    # so we explicitly import torch_dipu to guarantees that torch is patched by dipu
    # import torch_dipu
    import torch_npu
    from torch_npu.contrib import transfer_to_npu


_init()
