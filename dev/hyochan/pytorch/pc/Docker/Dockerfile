# ✅ CUDA 11.7 + PyTorch 1.13.1 (Windows는 linux/amd64로 대체 예시)
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    python3 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip

RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip3 install -r requirements.txt

CMD ["bash"]
