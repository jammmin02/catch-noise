#!/bin/bash

# ==============================
# Jetson 실시간 시각화용 Docker 실행 스크립트
# ==============================

IMAGE_NAME="jetson-noise-trt"
CONTAINER_NAME="jetson-noise-runner"
DOCKERFILE_PATH="dev/jungmin/jetson/Dockerfile"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ✅ X11 권한 부여
xhost +local:root

# ✅ 이미지 없으면 빌드
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
  echo "[INFO] Docker image not found. Building..."
  docker build -t $IMAGE_NAME -f "$PROJECT_ROOT/$DOCKERFILE_PATH" "$PROJECT_ROOT"
else
  echo "[INFO] Docker image already exists."
fi

# ✅ 컨테이너 실행 중인지 확인
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
  if docker inspect -f '{{.State.Running}}' $CONTAINER_NAME | grep -q true; then
    echo "[INFO] Container is already running."
  else
    echo "[INFO] Starting existing container..."
    docker start -ai $CONTAINER_NAME
  fi
else
  echo "[INFO] Running new container..."
  docker run -it --gpus all \
    --name $CONTAINER_NAME \
    --device /dev/snd \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume "$PROJECT_ROOT:/app" \
    --workdir /app/dev/jungmin/jetson \
    $IMAGE_NAME
fi
