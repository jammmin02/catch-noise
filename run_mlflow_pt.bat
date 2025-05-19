@echo off
setlocal enabledelayedexpansion

:: 경로 및 환경 변수 설정
set PROJECT_ROOT=%cd%
set CONTAINER_NAME=noise-jetson
set IMAGE_NAME=jetson-audio-rt
set DOCKERFILE_PATH=docker
set MLFLOW_TRACKING_URI=http://210.101.236.174:5000

:: Docker 이미지 확인 및 빌드
docker image inspect %IMAGE_NAME% >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Docker image not found. Building image...
    docker build -t %IMAGE_NAME% %DOCKERFILE_PATH%
) ELSE (
    echo [INFO] Docker image already exists.
)

:: 컨테이너 확인 및 생성 (❗ bash만 실행)
docker inspect %CONTAINER_NAME% >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Container not found. Creating new container...
    docker run -it -d --name %CONTAINER_NAME% ^
        --runtime nvidia ^
        --device /dev/snd ^
        --device /dev/input ^
        --privileged ^
        -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
        -v "%PROJECT_ROOT%":/app ^
        -w /app ^
        -p 5000:5000 ^
        %IMAGE_NAME% bash
) ELSE (
    echo [INFO] Container already exists.
)

:: 컨테이너 실행 상태 확인
docker inspect -f "{{.State.Running}}" %CONTAINER_NAME% | findstr "true" >nul
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Container is stopped. Starting it now...
    docker start %CONTAINER_NAME%
) ELSE (
    echo [INFO] Container is already running.
)

:: MLflow 중복 실행 방지
echo [INFO] Checking MLflow process...
docker exec %CONTAINER_NAME% pgrep -f "mlflow server" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] MLflow not running. Starting MLflow...
    docker exec -d %CONTAINER_NAME% bash -c "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///app/mlruns --default-artifact-root file:///app/mlartifacts"
) ELSE (
    echo [INFO] MLflow is already running.
)

:: bash 쉘 접속
echo [INFO] Connecting to container shell...
docker exec -it %CONTAINER_NAME% bash
