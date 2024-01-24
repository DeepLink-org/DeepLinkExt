def _init():
    # deeplink_ext is developed based on dipu
    # so we explicitly import torch_dipu to guarantees that torch is patched by dipu
    import torch_dipu


_init()
