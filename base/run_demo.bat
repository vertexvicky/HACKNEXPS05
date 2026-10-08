@echo off
title Single Cam Video Intelligence AI Server
echo =====================================================================
echo Launching Single Cam Video Intelligence Engine (CUDA + GenAI)
echo =====================================================================
echo.
cd /d %~dp0

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in base\venv.
    pause
    exit /b 1
)

echo Launching Uvicorn Server at http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
pause
