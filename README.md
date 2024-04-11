# DeepLinkExt
该仓库用提供了一套国产硬件进行大型模型训练、微调、推理的解决方案。对下调用[DIOPI](https://github.com/DeepLink-org/DIOPI)支持的大模型算子（如flash_attention和rms_norm），对上承接大模型训练、微调、推理框架；并对应提供了一套大模型算子的组合来实现。

DIOPI的具体内容请参考[DIOPI INTRODUCTION](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIOPI/Introduction.html)
目前支持的框架以及对应的算子可以查看[框架算子](https://github.com/DeepLink-org/DeepLinkExt/tree/main/deeplink_ext).

## Install
DeepLinkExt依赖deeplink.framework/dipu，需要先完成dipu的编译安装，具体请参考[dipu quick_start](https://deeplink.readthedocs.io/zh-cn/latest/doc/DIPU/quick_start.html)
完成dipu的编译后，请参考如下代码，设置必要的环境变量。
```bash
export PYTHONPATH=$WORKDIR/deeplink.framework/dipu/:$PYTHONPATH
export DIPU_ROOT=$WORKDIR/deeplink.framework/dipu/torch_dipu
export DIOPI_PATH=$WORKDIR/deeplink.framework/dipu/third_party/DIOPI/proto
export VENDOR_INCLUDE_DIRS=${PATH_TO_VENDOR_INCLUDE} # 底层软件栈的include路径，例如/usr/local/Ascend/ascend-toolkit/latest/include

```

完成上述准备工作后，使用如下命令即可安装DeepLinkExt

### 1. inplace安装

```
cd $WORKDIR/DeepLinkExt
python3 setup.py build_ext --inplace
```
### 2. 安装到指定目录

```bash
cd $WORKDIR/DeepLinkExt
pip install -e . -t $TARGET_INSTALL_DIR
```

## Usage
以InternEvo、LightLLM大模型训练框架为例，参考如下代码，即可实现在训练/推理时使用DeepLinkExt的扩展算子。
### InternEvo
DeepLinkExt已完全接入InternEvo，在完成DeepLinkExt的编译安装后，将其添加到PYTHONPATH，使用InternEvo进行训练即可。
### LightLLM
对于LightLLM，在启动推理的脚本中，需要添加如下代码，即可实现使用DeepLinkExt扩展算子。
```python
import deeplink_ext.patch_lightllm.py
import lightllm
```