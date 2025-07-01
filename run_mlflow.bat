@echo off
REM =============================================
REM MLflow 서버용 컨테이너 실행 (팀장만 사용)
REM =============================================

docker ps --format "{{.Names}}" | findstr /i noise-mlflow >nul
if %errorlevel%==0 (
    echo [INFO] MLflow 컨테이너가 이미 실행 중입니다.
) else (
    docker ps -a --format "{{.Names}}" | findstr /i noise-mlflow >nul
    if %errorlevel%==0 (
        echo [INFO] MLflow 컨테이너가 존재하지만 중지 상태입니다. 다시 시작합니다.
        docker start noise-mlflow
    ) else (
        echo [INFO] MLflow 컨테이너가 없으므로 새로 생성합니다.
        docker compose up -d mlflow
    )
)

echo [INFO] MLflow 서버가 실행 중입니다: http://210.101.236.174:5000
pause
