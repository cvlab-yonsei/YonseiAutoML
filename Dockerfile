FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
LABEL maintainer="hyunjulee@yonsei.ac.kr"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Core deps
RUN apt-get update && apt-get install -y \
    git vim zsh tmux htop curl wget locales tzdata \
    cmake build-essential gcc g++ gfortran \
    openmpi-bin libopenmpi-dev \
    ffmpeg libsm6 libxext6 libgtk2.0-dev \
    libpng-dev libfreetype6-dev libjpeg8-dev xdg-utils \
    libnss3 libxkbfile1 libsecret-1-0 \
    libgtk-3-0 libxss1 libasound2 libxtst6 libglib2.0-0 libdrm2 \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8 && update-locale
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Python deps
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN pip3 install -U pip setuptools wheel

# PyTorch 1.8.1 (CUDA 11.1)
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Extra packages (TF 제거, 최신화 가능)
RUN pip3 install \
    numpy==1.22.4 \
    scipy==1.7.3 \
    pandas==1.3.5 \
    matplotlib==3.5.3 \
    scikit-learn==1.0.2 \
    tqdm==4.65.0 \
    ptflops==0.7 \
    easydict==1.10 \
    einops \
    opencv_python==4.7.0.72 \
    thop==0.1.1.post2209072238 \
    timm==0.9.2 \
    tensorboard \
    ipython jupyter graphviz

# Horovod
ENV HOROVOD_WITH_PYTORCH=1
ENV HOROVOD_WITH_GLOO=1
ENV HOROVOD_GPU_OPERATIONS=NCCL
RUN pip3 install horovod==0.28.0

WORKDIR /data1/hyunju
CMD ["/bin/bash"]
