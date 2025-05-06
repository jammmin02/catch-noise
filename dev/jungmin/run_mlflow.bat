@echo off
REM 🔧 Move to the current script directory
cd /d %~dp0
cd ../..

REM 📌 Save project root path
set PROJECT_ROOT=%cd%

REM 📦 Docker settings
set IMAGE_NAME=tf-noise-preprocess
set CONTAINER_NAME=noise-runner
set DOCKERFILE=dev/jungmin/Docker/Dockerfile

REM 🔍 Check if the container already exists
docker ps -a --format "{{.Names}}" | findstr /i %CONTAINER_NAME% > nul
if %errorlevel%==0 (
    REM 컨테이너 존재함 → 실행 중인지 확인
    docker inspect -f "{{.State.Running}}" %CONTAINER_NAME% | findstr "true" > nul
    if %errorlevel%==0 (
        echo.
        echo ⚠️ Container "%CONTAINER_NAME%" is already running.
        echo 🔗 MLflow UI might already be available at: http://localhost:5000
        goto end
    ) else (
        echo.
        echo 🔄 Starting existing container "%CONTAINER_NAME%"...
        docker start -ai %CONTAINER_NAME%
        goto end
    )
)

echo.
echo [Step 1/3] Building Docker image...
docker build -t %IMAGE_NAME% -f %DOCKERFILE% %PROJECT_ROOT%

echo.
echo [Step 2/3] Running new Docker container with MLflow UI and bash shell...
docker run --name %CONTAINER_NAME% -it ^
-p 5000:5000 ^
-v %PROJECT_ROOT%:/app ^
-v %PROJECT_ROOT%\mlruns:/app/mlruns ^
-w /app ^
%IMAGE_NAME% ^
bash -c "mlflow ui --host 0.0.0.0 --port 5000 & exec bash"

:end
echo.
echo ✅ Script finished.
