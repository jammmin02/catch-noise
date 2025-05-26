# ✅ Python 3.10 기반 이미지
FROM python:3.10-slim

# ✅ 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean

# ✅ 작업 디렉토리 설정
WORKDIR /app

# ✅ 팀 MLflow 서버 URI 환경 변수
ENV MLFLOW_TRACKING_URI=http://210.101.236.174:5000

# ✅ 프로젝트 전체 복사
COPY . /app

# ✅ requirements 설치 (tensorflow 포함)
RUN pip install --upgrade pip && \
    pip install --force-reinstall --no-cache-dir -r docker/requirements.txt

# ✅ 기본 명령어는 bash
CMD ["bash"]
