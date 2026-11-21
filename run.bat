@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo [ERROR] The project environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
if not exist "%ROOT%\web\node_modules\.bin\vite.cmd" (
  echo [ERROR] Web dependencies are missing. Run setup.bat first.
  pause
  exit /b 1
)
"%PYTHON%" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] This environment must use Python 3.11. Run setup.bat after installing CPython 3.11 x64.
  pause
  exit /b 1
)

start "Clip Studio API" /D "%ROOT%" "%PYTHON%" -m uvicorn api.server:app --host 127.0.0.1 --port 8000
start "Clip Studio Web" /D "%ROOT%\web" cmd /k "npm run dev -- --host 127.0.0.1"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5173"
