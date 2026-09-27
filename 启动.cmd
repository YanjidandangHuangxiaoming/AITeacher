@echo off
rem ---------------------------------------------------------------
rem  ASCII ONLY. Do not put non-ASCII text in this file.
rem  cmd.exe misparses UTF-8 batch files after chcp 65001: its byte
rem  offset into the file desyncs and commands get split mid-word.
rem  The Chinese menu lives in scripts/launcher.py instead.
rem ---------------------------------------------------------------
chcp 65001 >nul
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo [ERROR] Virtual environment not found:
    echo         %PY%
    echo.
    echo         Run this in the project folder first:
    echo             python -m venv .venv
    echo             .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PY%" "%~dp0scripts\launcher.py"

rem Keep the window open if python died with an error.
if errorlevel 1 pause
