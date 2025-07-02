FROM nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3

# 필수 패키지 + 추가 라이브러리 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    # 마이크 인식을 위한 라이브러리 설치
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    alsa-utils \
    # 시각화를 위한 패키지 설치
    python3-tk \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app
COPY . /app

# pip 최신화
RUN pip3 install --upgrade pip

# 개별 패키지 설치
RUN pip3 install sounddevice onnxruntime

# requirements.txt 설치
RUN pip3 install --no-cache-dir --ignore-installed -r test/docker/requirements.txt


# 컨테이너 진입 시 bash
CMD ["bash"]

