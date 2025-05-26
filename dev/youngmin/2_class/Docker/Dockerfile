# ✅ TensorFlow 2.15.0 기반 이미지
FROM tensorflow/tensorflow:2.15.0

# ✅ 시스템 패키지 설치 (ffmpeg 포함)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean

# ✅ 🔥 충돌 방지: 시스템에 깔린 blinker 제거
RUN apt-get remove -y python3-blinker || true

# ✅ 작업 디렉토리 설정
WORKDIR /app

# ✅ 프로젝트 전체 복사 (run.sh, requirements.txt 포함)
COPY . /app

# ✅ pip 업그레이드 및 requirements 설치
RUN pip install --upgrade pip
RUN pip install --break-system-packages -r requirements.txt

# ✅ 기본 실행 명령어는 bash 셸
CMD ["bash"]
