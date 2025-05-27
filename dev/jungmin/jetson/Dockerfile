# ================================
# Jetson Nano 시각화 지원 Dockerfile
# ================================
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# ✅ 필수 시스템 패키지
RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    portaudio19-dev \
    ffmpeg \
    python3-tk \
    libgl1-mesa-glx \
    x11-utils \
    x11-xserver-utils \
    && apt-get clean

# ✅ Python 패키지 설치
RUN pip3 install --upgrade pip && \
    pip3 install \
    numpy \
    sounddevice \
    librosa \
    matplotlib \
    onnx \
    onnxruntime \
    pycuda

# ✅ 작업 디렉토리
WORKDIR /app

# ✅ 코드 복사
COPY . /app

# ✅ 기본 실행 생략 (bash 진입 후 직접 실행 추천)
