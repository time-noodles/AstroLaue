@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo   AstroLaue: Real-time Live Watcher Mode
echo   (Auto-detect new CSV + Screen capture)
echo ========================================================

:: 1. Search for Python in local virtual environment
set "PY_CMD="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
    goto :python_found
)

:: 2. Search common Conda installation paths (py312 env)
for %%D in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "C:\miniconda3"
    "C:\Miniconda3"
    "C:\anaconda3"
    "C:\Anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if exist "%%~D\envs\py312\python.exe" (
        set "PY_CMD=%%~D\envs\py312\python.exe"
        set "PATH=%%~D\envs\py312;%%~D\envs\py312\Scripts;%%~D\envs\py312\Library\bin;!PATH!"
        goto :python_found
    )
)

:: 3. Search other possible Conda env names (astrolaue, base)
for %%E in (astrolaue base) do (
    for %%D in (
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\Miniconda3"
        "%USERPROFILE%\anaconda3"
        "%USERPROFILE%\Anaconda3"
        "C:\miniconda3"
        "C:\Miniconda3"
    ) do (
        if exist "%%~D\envs\%%E\python.exe" (
            set "PY_CMD=%%~D\envs\%%E\python.exe"
            set "PATH=%%~D\envs\%%E;%%~D\envs\%%E\Scripts;%%~D\envs\%%E\Library\bin;!PATH!"
            goto :python_found
        )
    )
)

:: 4. Check if conda command is available in PATH
where conda >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=call conda run --no-capture-output -n py312 python"
    goto :python_found
)

:: 5. Check if standard python command in PATH works
where python >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :python_found
)

:: Python not found error
echo.
echo [ERROR] Python 3.12 or Conda environment was not found.
echo Please verify that Miniconda/Anaconda with env 'py312' is installed.
echo.
goto :script_end

:python_found
echo   Python: !PY_CMD!
echo ========================================================
echo   Auto-Watch: Monitoring raw CSV folder for auto-restore
echo   Console   : [C] Re-analyze latest CSV  /  [Enter] Screen capture
echo               [A] Screen alignment       /  [Q] Quit
echo   Hotkey    : [F9] Screen capture        /  [F10] Alignment
echo ========================================================
echo.

:: Execute main.py in live watcher mode
call !PY_CMD! main.py --live --output ./results_live/ --show-diagnostic --export-transparent %*

if errorlevel 1 (
    echo.
    echo [ERROR] AstroLaue terminated with error code: !ERRORLEVEL!
)

:script_end
echo.
echo ========================================================
echo   AstroLaue session ended.
echo ========================================================
pause
