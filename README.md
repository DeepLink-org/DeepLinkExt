# DeepLinkExt
在大模型训练框架中，常常会出现一些torch暂未实现的扩展算子，例如flash_attention、rms_norm等。  
本仓库用于在大模型训练框架适配国产硬件的过程中，对扩展算子进行替换，如果有对应的DIOPI实现，则直接使用，否则将其替换为若干个小算子组合实现。  
DIOPI的具体内容请参考[DIOPI INTRODUCTION](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIOPI/Introduction.html)  
目前支持的框架以及对应的算子可以查看DeepLink/deeplink_ext文件夹.

## Install
安装依赖deeplink/dipu，需要先完成dipu编译安装，具体请参考[dipu quick_start](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIPU/quick_start.html)
完成dipu的编译后，需要设置一些必要的环境变量
```bash
export PYTHONPATH=$WORKDIR/deeplink.framework/dipu/:$PYTHONPATH
export DIPU_ROOT=$WORKDIR/deeplink.framework/dipu/torch_dipu
export DIOPI_PATH=$WORKDIR/deeplink.framework/dipu/third_party/DIOPI/proto
export VENDOR_INCLUDE_DIRS=${PATH_TO_VENDOR_INCLUDE} # 底层软件栈的include路径，例如/usr/local/Ascend/ascend-toolkit/latest/include
```

完成上述准备后，使用如下命令即可安装DeepLinkExt
```bash
cd $WORKDIR/DeepLinkExt
pip install -e .
```

## Usage
以InternEvo和LightLLM大模型训练框架为例，参考如下代码，即可实现在训练时使用DeepLinkExt的扩展算子。
### InternEvo

```python
import deeplink_ext.internevo_ops
import internlm
```
### LightLLM

```python
import deeplink_ext.patch_lightllm.py
import lightllm
```