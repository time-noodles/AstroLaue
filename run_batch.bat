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

set "BATCH_DIR=share"
if not "%~1"=="" set "BATCH_DIR=%~1"

echo ========================================================
echo   AstroLaue: Batch Deblur (share)
echo   Target: %BATCH_DIR%
echo   Python: %PY_CMD%
echo ========================================================

%PY_CMD% main.py --batch-dir "%BATCH_DIR%" --output ./results_batch/ --export-transparent --validate-physics

if not errorlevel 1 (
    echo.
    echo [OK] Batch completed! Opening results folder...
    start "" "results_batch"
) else (
    echo.
    echo [ERROR] Batch processing failed.
)

echo.
pause
