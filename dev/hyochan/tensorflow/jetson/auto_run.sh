#!/bin/bash

# 현재 스크립트 위치 기준으로 프로젝트 루트로 이동 (상위 2단계)
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# 도커 변수 설정
IMAGE_NAME="noise-preprocess"
CONTAINER_NAME="noise-runner"
DOCKERFILE="hyochan/jetson_predict/Docker/Dockerfile"

# 컨테이너가 이미 실행 중인지 확인
if docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep true > /dev/null; then
    exit 0
fi

# 도커 이미지 빌드
echo "Building Docker image..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$PROJECT_ROOT"

# 컨테이너 실행
echo "Running container..."
docker run --gpus all --name "$CONTAINER_NAME" -it \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/snd:/dev/snd \
    -p 5000:5000 \
    -v "$PROJECT_ROOT":/app \
    -w /app \
    "$IMAGE_NAME" \
    bash -i


