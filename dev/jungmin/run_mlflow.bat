@echo off
REM 🔧 현재 스크립트 위치로 이동
cd /d %~dp0
cd ../..

REM 📌 프로젝트 루트 경로 저장
set PROJECT_ROOT=%cd%

REM 📦 Docker 설정
set IMAGE_NAME=tf-noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=dev/jungmin/Docker/Dockerfile

echo.
echo 📦 [1/3] Docker 이미지 빌드 중...
docker build -t %IMAGE_NAME% -f %DOCKERFILE% %PROJECT_ROOT%

echo.
echo 🧼 [2/3] 기존 컨테이너 제거 중 (있다면)...
docker rm -f %CONTAINER_NAME% > nul 2>&1

echo.
echo 🐳 [3/3] Docker 컨테이너 실행 중 (MLflow UI 자동 실행)...
docker run --name %CONTAINER_NAME% -it --rm ^
-p 5000:5000 ^
-v %PROJECT_ROOT%:/app ^
-v %PROJECT_ROOT%\mlruns:/app/mlruns ^
-w /app ^
%IMAGE_NAME% ^
bash -c "mlflow ui --host 0.0.0.0 --port 5000"

echo.
echo 🌐 MLflow UI: http://localhost:5000 에서 확인하세요!