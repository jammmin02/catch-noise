# ✅ Python 3.6.9 베이스 이미지 사용
FROM python:3.6.9-slim

# ✅ 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && apt-get clean

# ✅ 작업 디렉토리 설정
WORKDIR /app

# ✅ 프로젝트 전체 복사
COPY . /app

# ✅ pip 업그레이드 및 requirements 설치
RUN pip install --upgrade pip

# ✅ PyTorch 1.10.0 (CPU 버전) 설치
RUN pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# ✅ 나머지 의존성 설치
RUN pip install -r hyochan/pytorch_2class/Docker/requirements.txt

# ✅ 기본 실행 명령어
CMD ["bash"]
