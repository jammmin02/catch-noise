<<<<<<< HEAD
# ✅ 베이스 이미지 (GPU 필요 없으면 cpu 버전도 가능)
FROM tensorflow/tensorflow:2.13.0

# ✅ 시스템 패키지 설치
=======
# ✅ TensorFlow 2.13.0 기반 이미지
FROM tensorflow/tensorflow:2.13.0

# ✅ 시스템 패키지 설치 (ffmpeg 포함)
>>>>>>> origin/main
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean

# ✅ 작업 디렉토리 설정
WORKDIR /app
<<<<<<< HEAD
COPY . /app

# ✅ pip 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r dev/jungmin/Docker/requirements.txt

# ✅ 실행 명령어: bash 셸로 진입
=======

# ✅ 프로젝트 전체 복사 (run.sh, requirements.txt 포함)
COPY . /app

# ✅ pip 업그레이드 및 requirements 설치
RUN pip install --upgrade pip
RUN pip install -r dev/jungmin/Docker/requirements.txt

# ✅ 기본 실행 명령어는 bash 셸
>>>>>>> origin/main
CMD ["bash"]