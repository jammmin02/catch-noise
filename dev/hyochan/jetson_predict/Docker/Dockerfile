# ✅ NVIDIA에서 제공하는 Jetson용 TensorFlow 사전 설치 이미지 사용 (JetPack 4.6.4 대응)
FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

# ✅ 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ pip 업그레이드
RUN pip install --upgrade pip

# ✅ librosa 관련 필수 의존성 설치
RUN pip install numba==0.50.1 llvmlite==0.31.0

# ✅ 작업 디렉토리 설정
WORKDIR /app

# ✅ 코드 전체 복사
COPY . /app

# ✅ requirements.txt 설치
RUN pip install --no-cache-dir --ignore-installed -r hyochan/jetson/Docker/requirements.txt

# ✅ 기본 실행 명령어
CMD ["bash"]
