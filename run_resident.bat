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
echo   AstroLaue: Resident Hotkey Watcher Mode
echo   Python: %PY_CMD%
echo ========================================================
echo   Hotkey: [F9] or [Ctrl+Shift+L] to capture and deblur
echo   Console: Press [Enter] to capture, [Q] to quit
echo ========================================================
echo.

%PY_CMD% main.py --watch --output ./results_live/ --show-diagnostic --export-transparent %*

pause
