@echo off
REM 🔧 현재 스크립트 위치로 이동
cd /d %~dp0
cd ../..

REM 프로젝트 루트 경로 저장
set PROJECT_ROOT=%cd%

REM Docker 설정
set IMAGE_NAME=tf-noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=docker/Dockerfile

REM MLflow 서버 URI (정민이 고정 IP, 수정 불필요)
set MLFLOW_TRACKING_URI=http://210.101.236.174:5000

REM 컨테이너가 이미 존재하는지 확인
docker ps -a --format "{{.Names}}" | findstr /i %CONTAINER_NAME% > nul
if %errorlevel%==0 (
    REM 컨테이너 존재 → 실행 중인지 확인
    docker inspect -f "{{.State.Running}}" %CONTAINER_NAME% | findstr "true" > nul
    if %errorlevel%==0 (
        echo.
        echo  Container "%CONTAINER_NAME%" is already running.
        echo  MLflow UI might already be available at: %MLFLOW_TRACKING_URI%
        goto end
    ) else (
        echo.
        echo Starting existing container "%CONTAINER_NAME%"...
        docker start -ai %CONTAINER_NAME%
        goto end
    )
)

echo.
echo [Step 1/3] Building Docker image...
docker build --build-arg MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% -t %IMAGE_NAME% -f %DOCKERFILE% %PROJECT_ROOT%

echo.
echo [Step 2/3] Running new Docker container with MLflow UI and bash shell...
docker run --name %CONTAINER_NAME% -it ^
-p 5000:5000 ^
-v %PROJECT_ROOT%:/app ^
-v %PROJECT_ROOT%\mlruns:/app/mlruns ^
-w /app ^
-e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
%IMAGE_NAME% ^
bash -c "mlflow ui --host 0.0.0.0 --port 5000 & exec bash"

:end
echo.
echo MLflow UI is available at: %MLFLOW_TRACKING_URI%
