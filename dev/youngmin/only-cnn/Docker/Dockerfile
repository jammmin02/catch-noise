# 베이스 이미지: PyTorch + Python3.8
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# 기본 설정
ENV DEBIAN_FRONTEND=noninteractive

# 필요한 리눅스 패키지 설치
RUN apt-get update && \
    apt-get install -y wget ffmpeg libsndfile1-dev sox && \
    apt-get clean

# pip 업그레이드 및 Python 패키지 설치
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# 작업 디렉토리 설정
WORKDIR /workspace
