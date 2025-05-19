@echo off
REM 프로젝트 루트 경로 직접 설정 
set PROJECT_ROOT=C:\Users\USER\team-noise-ai-project
set PROJECT_ROOT_UNIX=C:/Users/USER/team-noise-ai-project
SET CONTAINER_NAME=noise-jetson
SET IMAGE_NAME=jetson-audio-rt
SET DOCKERFILE_PATH=docker
SET MLFLOW_TRACKING_URI=http://210.101.236.174:5000

REM [이미지 존재 여부 확인]
docker image inspect %IMAGE_NAME% >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Docker image not found. Building image...
    docker build -t %IMAGE_NAME% %DOCKERFILE_PATH%
) ELSE (
    echo Docker image already exists.
)

REM [컨테이너 존재 여부 확인]
docker inspect %CONTAINER_NAME% >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Container not found. Creating new container...

    REM 컨테이너가 없으면 새로 생성하고 백그라운드로 실행
    docker run -it -d --name %CONTAINER_NAME% ^
        --runtime nvidia ^
        --net=host ^
        --device /dev/snd ^
        --device /dev/input ^
        --privileged ^
        -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
        -v "%cd%":/app ^
        -w /app ^
        %IMAGE_NAME%
) ELSE (
    echo Container already exists.

    REM [컨테이너 실행 중인지 확인]
    docker inspect -f "{{.State.Running}}" %CONTAINER_NAME% | findstr "true" >nul
    IF %ERRORLEVEL% NEQ 0 (
        echo Container is stopped. Starting it now...
        docker start %CONTAINER_NAME%
    ) ELSE (
        echo Container is already running.
    )
)

REM [bash 셸 자동 진입]
echo Launching bash shell...
docker exec -it %CONTAINER_NAME% bash

REM 창이 꺼지지 않도록 대기
pause