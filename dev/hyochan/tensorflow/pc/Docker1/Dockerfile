# CUDA 11.8 + TensorFlow 2.11 (x86_64 용)
FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r hyochan/tensorflow/pc/Docker/requirements.txt

CMD ["bash"]
