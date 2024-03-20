# Copyright (c) 2024, DeepLink.

try:
    from .deeplink import adamw, DeeplinkAdamW
except:
    raise ImportError("[deeplink_ext] adamw is not implemented in diopi.\n")
