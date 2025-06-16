# 베이스 이미지
FROM tensorflow/tensorflow:2.13.0

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && apt-get clean

# 작업 디렉토리
WORKDIR /app

# 프로젝트 전체 복사
COPY . /app

# 의존성 설치
RUN pip install --upgrade pip
RUN pip install -r dev/youngmin/Docker/requirements.txt

# 디폴트 명령어 (원하면 바꿔도 됨)
CMD ["bash"]
