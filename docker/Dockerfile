# ✅ Python 3.8 기반 이미지 (slim 버전으로 용량 최소화)
FROM python:3.8-slim

# ✅ 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && apt-get clean

# ✅ 작업 디렉토리
WORKDIR /app
COPY . /app

# ✅ pip 업그레이드 및 requirements 설치
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 기본 명령어
CMD ["bash"]