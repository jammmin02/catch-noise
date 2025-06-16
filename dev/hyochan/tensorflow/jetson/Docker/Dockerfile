# NVIDIA에서 제공하는 Jetson용 TensorFlow 사전 설치 이미지 사용 (JetPack 4.6.4 대응)
FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

# 1. 필수 패키지 설치 (llvm-9 포함 + sounddevice용 라이브러리 포함)
RUN apt-get update && apt-get install -y \
    llvm-9 \
    llvm-9-dev \
    llvm-9-tools \
    ffmpeg \
    git \
    build-essential \
    libportaudio2 \
    libsndfile1 \
    python3-pip \
    python3-dev \
    alsa-utils \
    python3-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. pip 업그레이드
RUN pip3 install --upgrade pip

# 3. LLVM 경로 지정 후 numba + llvmlite 설치
ENV LLVM_CONFIG=/usr/bin/llvm-config-9
RUN pip3 install numba==0.50.1 llvmlite==0.33.0

# 4. 추가 파이썬 패키지 설치
RUN pip3 install sounddevice onnxruntime

# 작업 디렉토리 설정
WORKDIR /app

# 코드 전체 복사
COPY . /app

# requirements.txt 설치
RUN pip3 install --no-cache-dir --ignore-installed -r hyochan/jetson_predict/Docker/requirements.txt

# 기본 실행 명령어
CMD ["bash"]
