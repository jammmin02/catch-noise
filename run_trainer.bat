@echo off
REM =============================================
REM 실험용 고정 컨테이너 실행 + bash 진입 (팀 전체 사용)
REM =============================================

docker ps --format "{{.Names}}" | findstr /i noise-trainer >nul
if %errorlevel%==0 (
    echo [INFO] Trainer 컨테이너 실행 중 → bash 진입
    docker exec -it noise-trainer bash
) else (
    docker ps -a --format "{{.Names}}" | findstr /i noise-trainer >nul
    if %errorlevel%==0 (
        echo [INFO] Trainer 컨테이너는 존재하나 중지 상태 → 재시작 후 bash 진입
        docker start noise-trainer
        docker exec -it noise-trainer bash
    ) else (
        echo [INFO] Trainer 컨테이너가 없으므로 새로 생성 + bash 진입
        docker compose up -d trainer
        docker exec -it noise-trainer bash
    )
)

pause
