import sys
import os
cur_work_dir = os.path.dirname(__file__)
sys.path.append(cur_work_dir)
from .ext_apply.lightllm.mock_op import *