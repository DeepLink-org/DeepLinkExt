import sys
from .kernels import ascend as ascend_kernels
from .engine.devices import ascend as ascend_engine_device


sys.modules['lmdeploy.pytorch.kernels.ascend'] = ascend_kernels
sys.modules['lmdeploy.pytorch.engine.devices.ascend'] = ascend_engine_device
