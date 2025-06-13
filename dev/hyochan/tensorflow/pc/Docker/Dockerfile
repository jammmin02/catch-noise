# ✅ GPU 지원 TensorFlow 이미지 기반 (CUDA 11.8 포함)
FROM tensorflow/tensorflow:2.7.0-gpu

# ✅ 시스템 패키지 설치 (NVIDIA GPG 키 문제 해결 포함)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# ✅ pip 업그레이드
RUN pip install --upgrade pip

# ✅ 작업 디렉토리
WORKDIR /app

# ✅ 전체 복사
COPY . /app

# ✅ requirements 설치 (distutils 설치 우회)
RUN pip install --no-cache-dir --ignore-installed -r hyochan/tensorflow/pc/Docker/requirements.txt

# ✅ 기본 실행 명령어
CMD ["bash"]
