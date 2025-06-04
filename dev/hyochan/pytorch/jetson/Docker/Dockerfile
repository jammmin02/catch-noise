# ✅ Jetson Orin / JetPack 5.1.3 (L4T 35.3.1 기준)
FROM nvcr.io/nvidia/l4t-pytorch:r35.3.1-pth1.13-py3

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app
COPY . /app

# pip 최신화
RUN python3 -m pip install --upgrade pip

# requirements.txt 설치 (나머지 패키지)
RUN pip3 install -r requirements.txt

# 디폴트 명령어
CMD ["bash"]
