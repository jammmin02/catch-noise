@echo off
REM 현재 스크립트 위치로 이동
cd /d %~dp0
cd ../..

REM 🔧 프로젝트 루트 경로 (Windows → Unix 스타일도 변환)
set "PROJECT_ROOT=%cd%"
set "PROJECT_ROOT_UNIX=%PROJECT_ROOT:\=/%"

REM 🐳 Docker 설정
set "IMAGE_NAME=noise-preprocess"
set "CONTAINER_NAME=noise-runner"
set "DOCKERFILE=hyochan/jetson/Docker/Dockerfile"
set "MLFLOW_TRACKING_URI=http://210.101.236.174:5000"

REM 컨테이너가 실행 중이면 안내만 출력
docker inspect -f "{{.State.Status}}" %CONTAINER_NAME% 2>nul | findstr "running" >nul
if %errorlevel%==0 (
    echo ✅ Container already running: MLflow UI → %MLFLOW_TRACKING_URI%
    goto end
)

REM 도커 이미지 빌드
echo 🛠️ Building Docker image...
docker build -t %IMAGE_NAME% -f "%DOCKERFILE%" "%PROJECT_ROOT%"

REM 컨테이너 실행만 (bash 인터랙티브 모드 + 종료 시 자동 삭제)
echo 🚀 Running container...
docker run --rm --gpus all --name %CONTAINER_NAME% -it ^
    -p 5000:5000 ^
    -v "%PROJECT_ROOT_UNIX%":/app ^
    -w /app ^
    -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
    %IMAGE_NAME% ^
    bash -i

:end
echo 🌐 MLflow UI is available at: %MLFLOW_TRACKING_URI%
