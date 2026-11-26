@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"

echo.
echo ==================================================
echo   All-in-One AI Video Clipping Studio - setup
echo ==================================================
echo.

where py >nul 2>nul
if errorlevel 1 goto :python_launcher_missing
py -3.11 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if errorlevel 1 goto :python311_missing
echo [OK] CPython 3.11 x64 found.

where ffmpeg >nul 2>nul
if errorlevel 1 goto :ffmpeg_missing
where ffprobe >nul 2>nul
if errorlevel 1 goto :ffprobe_missing
echo [OK] FFmpeg and ffprobe found on PATH.

where node >nul 2>nul
if errorlevel 1 goto :node_missing
node -e "const [major,minor]=process.versions.node.split('.').map(Number); const ok=(major>22)||(major===22&&minor>=12)||(major===20&&minor>=19); process.exit(ok?0:1)" >nul 2>nul
if errorlevel 1 goto :node_version_missing
where npm >nul 2>nul
if errorlevel 1 goto :npm_missing
echo [OK] Supported Node.js and npm found.

where ollama >nul 2>nul
if errorlevel 1 (
  echo [OPTIONAL] Ollama was not found. Install it separately to use local models.
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>nul
  if errorlevel 1 (
    echo [OPTIONAL] Ollama is installed but its local service did not respond at 127.0.0.1:11434.
  ) else (
    echo [OK] Ollama local API responded.
  )
)

echo.
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment with Python 3.11...
  py -3.11 -m venv .venv
  if errorlevel 1 goto :venv_failed
)
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"
"%VENV_PY%" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if errorlevel 1 goto :venv_wrong_python

"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :pip_failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed

rem MediaPipe's metadata also installs opencv-contrib-python. Reapply the
rem pinned headless wheel last so cv2 resolves to the headless build.
"%VENV_PY%" -m pip install --no-deps --force-reinstall opencv-python-headless==4.11.0.86
if errorlevel 1 goto :pip_failed

pushd "%ROOT%\web"
call npm ci
if errorlevel 1 (
  popd
  goto :npm_install_failed
)
call npm run build
if errorlevel 1 (
  popd
  goto :npm_build_failed
)
popd

echo.
echo Probing the FFmpeg H.264 encoder selection...
"%VENV_PY%" -m core.encoder --check
if errorlevel 1 goto :encoder_probe_failed

echo.
echo [DONE] Setup completed. Run run.bat to launch the studio.
echo.
pause
exit /b 0

:python_launcher_missing
echo [ERROR] The Windows Python launcher 'py' was not found.
echo Install CPython 3.11 x64 and enable the Python launcher, then retry.
goto :failed

:python311_missing
echo [ERROR] CPython 3.11 x64 is required. Python 3.12+ is not supported by this pinned setup.
echo Install Python 3.11 x64 from https://www.python.org/downloads/ and retry.
goto :failed

:ffmpeg_missing
echo [ERROR] FFmpeg was not found on PATH. Install an FFmpeg build with H.264 support, then retry.
goto :failed

:ffprobe_missing
echo [ERROR] ffprobe was not found on PATH. Install it alongside FFmpeg, then retry.
goto :failed

:node_missing
echo [ERROR] Node.js was not found. Install Node.js 22.12+ or 20.19+ and retry.
goto :failed

:node_version_missing
echo [ERROR] Node.js 22.12+ or 20.19+ is required by the pinned Vite toolchain.
goto :failed

:npm_missing
echo [ERROR] npm was not found. Reinstall Node.js with npm enabled, then retry.
goto :failed

:venv_failed
echo [ERROR] Could not create the Python 3.11 virtual environment.
goto :failed

:venv_wrong_python
echo [ERROR] The existing .venv is not a 64-bit Python 3.11 environment.
echo Remove .venv and run setup.bat again with CPython 3.11 x64 installed.
goto :failed

:pip_failed
echo [ERROR] Python dependency installation failed. Review the pip output above.
goto :failed

:npm_install_failed
echo [ERROR] Web dependency installation failed. Review the npm output above.
goto :failed

:npm_build_failed
echo [ERROR] The production UI build failed. Review the npm output above.
goto :failed

:encoder_probe_failed
echo [ERROR] No usable H.264 encoder passed the runtime probe. Ensure libx264 is available.
goto :failed

:failed
echo.
echo Setup did not complete. Fix the reported prerequisite and run setup.bat again.
echo.
pause
exit /b 1
