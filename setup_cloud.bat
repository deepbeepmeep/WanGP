@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" cloud\modal\setup_modal.py %*
) else (
    python cloud\modal\setup_modal.py %*
)
if %ERRORLEVEL% neq 0 (
    pause
)
