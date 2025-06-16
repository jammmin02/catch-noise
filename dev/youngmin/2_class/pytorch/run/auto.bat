@echo off
chcp 65001
setlocal

set CONTAINER_NAME=noise_dev
set IMAGE_NAME=noise_image

echo [1] Docker 이미지 빌드 (최초 1회만 오래 걸림)
docker build -t %IMAGE_NAME% -f ..\Docker\Dockerfile.pc ..

echo [2] Docker 컨테이너 bash 진입
docker run -it --rm ^
    --name %CONTAINER_NAME% ^
    -v %cd%\..:/workspace ^
    %IMAGE_NAME% /bin/bash
