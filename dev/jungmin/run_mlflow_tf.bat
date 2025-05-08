@echo off
REM 🔧 프로젝트 루트 경로 직접 설정 (정민이 PC 기준)
set PROJECT_ROOT=C:\Users\USER\team-noise-ai-project
set PROJECT_ROOT_UNIX=C:/Users/USER/team-noise-ai-project
set IMAGE_NAME=tf-noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=docker/Dockerfile
set MLFLOW_TRACKING_URI=http://210.101.236.174:5000

REM 🔧 mlruns 디렉토리가 없으면 생성
if not exist "%PROJECT_ROOT%\mlruns" (
    echo [Info] Creating 'mlruns' directory...
    mkdir "%PROJECT_ROOT%\mlruns"
)

REM 🟢 컨테이너가 이미 실행 중인지 확인
docker inspect -f "{{.State.Running}}" %CONTAINER_NAME% 2>nul | findstr "true" >nul
if %errorlevel%==0 (
    echo [Info] Container "%CONTAINER_NAME%" is already running.
    echo [Info] Attaching to the container...
    docker exec -it %CONTAINER_NAME% bash
    goto end
)

REM 🔁 컨테이너가 존재하지만 중지 상태일 경우
docker ps -a --format "{{.Names}}" | findstr /i %CONTAINER_NAME% >nul
if %errorlevel%==0 (
    echo [Info] Restarting stopped container "%CONTAINER_NAME%"...
    docker start %CONTAINER_NAME%
    timeout /t 3 >nul
    docker exec -d %CONTAINER_NAME% bash -c "mlflow ui --host 0.0.0.0 --port 5000"
    docker exec -it %CONTAINER_NAME% bash
    goto end
)

REM 🛠️ 새로 이미지 빌드 및 컨테이너 생성
echo [Step 1/3] Building Docker image...
docker build -t %IMAGE_NAME% -f %DOCKERFILE% %PROJECT_ROOT%

echo [Step 2/3] Creating and starting the container...
docker run --name %CONTAINER_NAME% -it ^
    -p 5000:5000 ^
    -v %PROJECT_ROOT_UNIX%:/app ^
    -v %PROJECT_ROOT_UNIX%/mlruns:/app/mlruns ^
    -w /app ^
    -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
    %IMAGE_NAME% ^
    bash -c "mlflow ui --host 0.0.0.0 --port 5000 & bash"

:end
echo.
echo [Info] MLflow UI available at: %MLFLOW_TRACKING_URI%
pause