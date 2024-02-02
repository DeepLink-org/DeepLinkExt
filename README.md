# DeepLinkExt

基本思想仿照 cpp extension，不过会先在 python 层判断该融合算子的 diopi 实现没有（具体判断方法为，在 cpp 层进行 pybind 时，如果没有 diopi 实现，则不进行 pybind）。如果没有实现，则会在 python 层替换为 torch 的几个分离算子。

融合算子的 diopi 定义及实现放在 DIOPI 库里，本拓展库仅引用。

支持自动 patch InternLM 和 LightLLM 中用到的融合算子，将它们替换为 DIOPI 实现。

## Install

首先安装 DIPU，确保可以 `import torch_dipu`。然后在本目录下执行

```bash
pip install -e .
```

## Usage

### InternLM

```python
import deeplink_ext.patch_internlm
import internlm
```

### LightLLM

```python
import deeplink_ext.patch_lightllm
import lightllm
```
