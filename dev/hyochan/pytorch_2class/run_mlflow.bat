@echo off
REM 💡 현재 위치로 이동
cd /d %~dp0
cd ../..

REM ✅ 루트 경로 설정
set PROJECT_ROOT=%cd%
set "PROJECT_ROOT_UNIX=%PROJECT_ROOT:\=/%"

REM ✅ Docker 환경 변수 설정
set IMAGE_NAME=tf-noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=hyochan/pytorch_2class/Docker/Dockerfile
set MLFLOW_TRACKING_URI=http://210.101.236.174:5000

echo ================================================
echo 📍 프로젝트 루트: %PROJECT_ROOT%
echo 🐳 이미지 이름: %IMAGE_NAME%
echo 📂 도커파일: %DOCKERFILE%
echo 🌐 MLflow 서버: %MLFLOW_TRACKING_URI%
echo ================================================

REM ✅ 컨테이너 중지 상태 확인 후 재시작
docker inspect -f "{{.State.Status}}" %CONTAINER_NAME% 2>nul | findstr "exited" >nul
if %errorlevel%==0 (
    echo 🔁 중지된 컨테이너 발견 → 재시작 중...
    docker start -ai %CONTAINER_NAME%
    goto end
)

REM ✅ 컨테이너 실행 중이면 안내만
docker inspect -f "{{.State.Status}}" %CONTAINER_NAME% 2>nul | findstr "running" >nul
if %errorlevel%==0 (
    echo ✅ 이미 실행 중인 컨테이너: MLflow UI → %MLFLOW_TRACKING_URI%
    goto end
)

REM ✅ 새로 이미지 빌드
echo 🛠️ Docker 이미지 빌드 중...
docker build -t %IMAGE_NAME% -f %DOCKERFILE% %PROJECT_ROOT%

REM ✅ 컨테이너 실행 (bash 유지)
echo 🚀 컨테이너 실행 및 bash 진입...
docker run --name %CONTAINER_NAME% -it ^
    -p 5000:5000 ^
    -v %PROJECT_ROOT_UNIX%:/app ^
    -w /app ^
    -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
    %IMAGE_NAME% bash

:end
echo 🌐 MLflow UI is available at: %MLFLOW_TRACKING_URI%

# ✅ 컨테이너가 꺼지지 않게 유지
CMD ["tail", "-f", "/dev/null"]

