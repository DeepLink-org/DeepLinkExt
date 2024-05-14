FROM registry.cn-sh-01.sensecore.cn/lepton-trainingjob/ubuntu18.04-pytorch2.1.0-py38-npu-910b-7.0.0:v1.0.0-20240130

USER root
WORKDIR /root/workspace
ENV TZ=UTC-8

# 下载并安装toolkit, kernels, miniconda
RUN useradd HwHiAiUser \
        && wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC703/Ascend-cann-toolkit_8.0.RC1.alpha003_linux-x86_64.run \
        && chmod 777 Ascend-cann-toolkit_8.0.RC1.alpha003_linux-x86_64.run \
        && ./Ascend-cann-toolkit_8.0.RC1.alpha003_linux-x86_64.run --full --quiet \
        && rm Ascend-cann-toolkit_8.0.RC1.alpha003_linux-x86_64.run -rf \
        && wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC703/Ascend-cann-kernels-910b_8.0.RC1.alpha003_linux.run \
        && chmod 777 Ascend-cann-kernels-910b_8.0.RC1.alpha003_linux.run \
        && ./Ascend-cann-kernels-910b_8.0.RC1.alpha003_linux.run --install --quiet \
        && rm Ascend-cann-kernels-910b_8.0.RC1.alpha003_linux.run -rf \
        && wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.9.0-0-Linux-x86_64.sh \
        && bash Miniconda3-py39_23.9.0-0-Linux-x86_64.sh -b -p /opt/miniconda3 \
        && rm Miniconda3-py39_23.9.0-0-Linux-x86_64.sh \
        && export http_proxy=http://proxy.sensetime.com:3128 \
        && export https_proxy=http://proxy.sensetime.com:3128 \
        && wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash

# 将miniconda设置到PATH中
ENV PATH="/opt/miniconda3/bin:${PATH}"

# 创建dipu的conda环境
RUN conda init bash \
        && conda create -n dipu_dev_py39 --clone base

# 安装必要的python包
RUN /bin/bash -c " \
        source activate dipu_dev_py39 \
        && pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
        && pip install setuptools==65.7.0 deepspeed==0.9.2 numpy==1.23.0 python-dateutil==2.8.2 \
        cmake wheel pyyaml transformers sentencepiece protobuf pybind11 decorator scipy attrs \
        tensorboard==2.12.0"

# 升级git, 安装openssh
RUN /bin/bash -c " \
        source activate dipu_dev_py39 \
        && apt-get update \
        && apt-get upgrade git -y \
        && apt-get install openssh-server -y"

# 安装apex,deepspeed_npu,megatron,megatron_npu
RUN /bin/bash -c " \
        source activate dipu_dev_py39 \
        && export http_proxy=http://proxy.sensetime.com:3128 \
        && export https_proxy=http://proxy.sensetime.com:3128 \
        && git config --global http.postBuffer 524288000 \
        && git config --global http.lowSpeedLimit 0 \
        && git config --global http.lowSpeedTime 999999 \
        && git clone https://gitee.com/ascend/apex.git \
        && cd apex/ \
        && git checkout 452c8da \
        && bash scripts/build.sh --python=3.9 \
        && cd apex/dist/ \
        && pip install apex-0.1_ascend-cp39-cp39-linux_x86_64.whl \
        && cd ../../../ \
        && git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu \
        && cd deepspeed_npu/ \
        && pip install -e . \
        && cd ../ \
        && git clone https://github.com/NVIDIA/Megatron-LM.git \
        && cd Megatron-LM/ \
        && git checkout 285068c \
        && git clone https://gitee.com/ascend/Megatron-LM.git megatron_npu_adaptor \
        && cd megatron_npu_adaptor/ \
        && git checkout a8958a5 \
        && pip install -e . \
        && cd .. \
        && pip install -e ."

# 安装dipu
RUN /bin/bash -c " \
        source activate dipu_dev_py39 \
        && export http_proxy=http://proxy.sensetime.com:3128 \
        && export https_proxy=http://proxy.sensetime.com:3128 \
        && git config --global http.postBuffer 524288000 \
        && git config --global http.lowSpeedLimit 0 \
        && git config --global http.lowSpeedTime 999999 \
        && git clone https://github.com/DeepLink-org/deeplink.framework.git \
        && cd deeplink.framework \
        && git checkout 2f9cb9f \
        && git submodule update --init --recursive \
        && cd dipu \
        && cd third_party/DIOPI/ \
        && git checkout 773e6c7 \
        && cd ../.. \
        && bash scripts/ci/ascend/ci_ascend_script.sh build_dipu"

# 安装deeplink_ext
RUN /bin/bash -c " \
        source activate dipu_dev_py39 \
        && export http_proxy=http://proxy.sensetime.com:3128 \
        && export https_proxy=http://proxy.sensetime.com:3128 \
        && git config --global http.postBuffer 524288000 \
        && git config --global http.lowSpeedLimit 0 \
        && git config --global http.lowSpeedTime 999999 \
        && git clone https://github.com/DeepLink-org/DeepLinkExt.git \
        && cd DeepLinkExt \
        && git checkout 662e678 \
        && export VENDOR_INCLUDE_DIRS=/usr/local/Ascend/ascend-toolkit/latest/include \
        && export DIOPI_PATH=/root/workspace/deeplink.framework/dipu/third_party/DIOPI/proto \
        && export DIPU_ROOT=/root/workspace/deeplink.framework/dipu/torch_dipu \
        && pip install -e ."

RUN /bin/bash -c " \
        source activate dipu_dev_py39 \
        && export http_proxy=http://proxy.sensetime.com:3128 \
        && export https_proxy=http://proxy.sensetime.com:3128 \
        && git config --global http.postBuffer 524288000 \
        && git config --global http.lowSpeedLimit 0 \
        && git config --global http.lowSpeedTime 999999 \
        && git clone https://github.com/InternLM/InternEvo.git \
        && cd InternEvo \
        && git checkout b024cf4 "

# 把dipu设置到默认的PYTHONPATH中
RUN mkdir -p /opt/miniconda3/envs/dipu_dev_py39/etc/conda/activate.d \
        && echo "export PYTHONPATH=/root/workspace/deeplink.framework/dipu:\$PYTHONPATH" >> \
        /opt/miniconda3/envs/dipu_dev_py39/etc/conda/activate.d/env_vars.sh \
        && echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc \
        && echo "source /root/workspace/git-completion.bash" >> ~/.bashrc

RUN export http_proxy=http://proxy.sensetime.com:3128 \
        && export https_proxy=http://proxy.sensetime.com:3128 \
        && wget https://github.com/krallin/tini/releases/download/v0.19.0/tini \
        && chmod +x tini \
        && ./tini -s sleep inf &
