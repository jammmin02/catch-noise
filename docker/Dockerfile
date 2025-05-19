FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# GPG 키 추가 (NVIDIA 저장소용)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 libasound2-dev \
    libportaudio2 libportaudiocpp0 portaudio19-dev\
    libsm6 libxext6 libxrender-dev \
    git curl unzip cmake build-essential \
    python3-pip python3.6 python3.6-dev \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 링크
RUN ln -s /usr/bin/python3.6 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["/bin/bash"]
