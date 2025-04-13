@echo off
REM 💡 현재 경로: dev/jungmin/Docker
cd /d %~dp0

REM 🔼 두 단계 위로 올라가서 루트 폴더로 이동
cd ../..

REM ✅ 루트 경로 저장
set PROJECT_ROOT=%cd%

REM ✅ Docker 관련 설정
set IMAGE_NAME=tf-noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=dev/jungmin/Docker/Dockerfile

echo.
echo Building Docker image...
docker build -t %IMAGE_NAME% -f %DOCKERFILE% %PROJECT_ROOT%

echo.
echo Removing existing container (if any)...
docker rm -f %CONTAINER_NAME% > nul 2>&1

echo.
echo Starting new container and launching bash shell...
docker run --name %CONTAINER_NAME% -it --rm ^
-v %PROJECT_ROOT%:/app ^
-w /app ^
%IMAGE_NAME% bash