@echo off
setlocal
cd /d "%~dp0"

set "CONDA_DIR=%USERPROFILE%\Miniconda3"
if not exist "%CONDA_DIR%\envs\py312\python.exe" set "CONDA_DIR=%USERPROFILE%\miniconda3"
if not exist "%CONDA_DIR%\envs\py312\python.exe" set "CONDA_DIR=C:\Miniconda3"
if not exist "%CONDA_DIR%\envs\py312\python.exe" set "CONDA_DIR=C:\miniconda3"

if exist "%CONDA_DIR%\envs\py312\python.exe" (
    set "PY_CMD=%CONDA_DIR%\envs\py312\python.exe"
    set "PATH=%CONDA_DIR%\envs\py312;%CONDA_DIR%\envs\py312\Scripts;%CONDA_DIR%\envs\py312\Library\bin;%PATH%"
) else (
    set "PY_CMD=conda run --no-capture-output -n py312 python"
)

echo ========================================================
echo   AstroLaue: Screen / Active Window Capture
echo   Python: %PY_CMD%
echo ========================================================
echo   Capturing current screen or active window...
echo.

%PY_CMD% main.py --live --output ./results_live/ --show-diagnostic --export-transparent %*

if not errorlevel 1 (
    echo.
    echo [OK] Capture completed! Opening results folder...
    start "" "results_live"
) else (
    echo.
    echo [ERROR] Capture failed. Please verify the diffraction pattern is displayed on screen.
)

echo.
pause
