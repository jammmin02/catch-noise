@echo off
REM 현재 위치로 이동
cd /d %~dp0
cd ../../..

set PROJECT_ROOT=%cd%
set "PROJECT_ROOT_UNIX=%PROJECT_ROOT:\=/%"

set IMAGE_NAME=tf-noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=dev/hyochan/Docker/Dockerfile
set MLFLOW_TRACKING_URI=http://210.101.236.174:5000

REM 컨테이너 중지 상태이면 재시작
docker inspect -f "{{.State.Status}}" %CONTAINER_NAME% 2>nul | findstr "exited" >nul
if %errorlevel%==0 (
    echo 🔁 Stopped container found, restarting...
    docker start -ai %CONTAINER_NAME%
    goto end
)

REM 컨테이너 실행 중이면 안내만
docker inspect -f "{{.State.Status}}" %CONTAINER_NAME% 2>nul | findstr "running" >nul
if %errorlevel%==0 (
    echo ✅ Container already running: MLflow UI → %MLFLOW_TRACKING_URI%
    goto end
)

echo 🛠️ Building Docker image...
docker build -t %IMAGE_NAME% -f %DOCKERFILE% %PROJECT_ROOT%

echo 🚀 Running container and executing training script...
docker run --name %CONTAINER_NAME% -it ^
    -p 5000:5000 ^
    -v %PROJECT_ROOT_UNIX%:/app ^
    -w /app ^
    -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
    %IMAGE_NAME% ^
    bash -c "python dev/hyochan/2class_predict_model/train/keras_train_lstm_2class.py; exec bash"

:end
echo 🌐 MLflow UI is available at: %MLFLOW_TRACKING_URI%
