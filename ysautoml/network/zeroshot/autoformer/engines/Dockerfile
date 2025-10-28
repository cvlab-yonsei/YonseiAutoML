FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
MAINTAINER <junghyup.lee@yonsei.ac.kr>

# Update and install
RUN apt-get update
RUN apt-get install -y \
      git \
      vim \
      zsh \
      tmux \
      htop \
      curl \
      wget \
      locales

# Install language pack
RUN apt-get install -y language-pack-en
RUN locale-gen en_US.utf8
RUN update-locale
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install dev for cv2
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y libgtk2.0-dev
RUN pip install -U scikit-learn
RUN pip install scikit-image

RUN apt-get install -y libpng-dev
RUN apt-get install -y libfreetype6-dev
RUN apt-get install -y libjpeg8-dev
RUN pip install matplotlib

RUN pip install jupyter numpy scipy ipython pandas easydict opencv-python tensorflow torchsummary tensorboard
RUN pip install horovod[pytorch] ptflops timm thop einops graphviz
RUN apt-get install -y xdg-utils

#CleanUp
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

#Apex
WORKDIR /
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /apex/
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

CMD ["/bin/bash"]
