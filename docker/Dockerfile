# TensorFlow 2.13.0 기반 이미지
FROM tensorflow/tensorflow:2.13.0

# 시스템 패키지 설치 (ffmpeg 포함)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수로 팀 MLflow 서버 URI 고정
ENV MLFLOW_TRACKING_URI=http://210.101.236.174:5000

# 프로젝트 전체 복사
COPY . /app

# pip 업그레이드 및 requirements 설치
RUN pip install --upgrade pip
RUN pip install -r docker/requirements.txt

# 기본 실행 명령어는 bash 셸
CMD ["bash"]
