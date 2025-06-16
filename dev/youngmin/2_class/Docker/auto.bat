@echo off
chcp 65001 > nul
REM 현재 경로: dev/youngmin/2_class/Docker
cd /d %~dp0

REM Dockerfile과 requirements.txt가 있는 현재 디렉토리 저장
set DOCKER_DIR=%cd%

REM 루트 경로 저장 (전체 프로젝트)
cd ../..
set PROJECT_ROOT=%cd%

REM Docker 관련 설정
set IMAGE_NAME=noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=%DOCKER_DIR%\Dockerfile

echo.
echo [1/3] Docker 이미지 빌드 중...
docker build -t %IMAGE_NAME% -f "%DOCKERFILE%" "%DOCKER_DIR%"
if %errorlevel% neq 0 (
    echo 이미지 빌드 실패. run 생략됨.
    pause
    exit /b
)

echo.
echo [2/3] 기존 컨테이너 제거 (있다면)...
docker rm -f %CONTAINER_NAME% > nul 2>&1

echo.
echo [3/3] 새 컨테이너 실행 중 (bash 진입)...
docker run -it --rm ^
 -v %cd%:/workspace ^
 -p 5000:5000 -p 8888:8888 ^
 --name %CONTAINER_NAME% ^
 %IMAGE_NAME% bash

