# 데스크탑(x86_64)용 PyTorch 1.9.0 + CUDA 10.2
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.6 python3-pip python3.6-dev \
    ffmpeg libsndfile1 libasound2-dev \
    libsm6 libxext6 libxrender-dev \
    git curl unzip cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 기본 설정
RUN ln -s /usr/bin/python3.6 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# requirements 복사 및 설치
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["/bin/bash"]
