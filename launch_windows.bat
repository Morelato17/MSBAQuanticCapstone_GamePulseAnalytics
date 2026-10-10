@echo off
setlocal

rem Double-click launcher for Windows.
rem %~dp0 is the directory containing this file, including the trailing slash.

cd /d "%~dp0"
set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS=%~dp0requirements.txt"
set "REQUIREMENTS_STAMP=%VENV_DIR%\.requirements.txt"

echo GamePulse Analytics - Community Health Dashboard
echo =================================================
echo.

if not exist "%VENV_PYTHON%" (
    echo First launch: creating the Python environment...

    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo ERROR: Python 3 is not installed or is not available on PATH.
            echo Install Python 3.12 from https://www.python.org/downloads/ and try again.
            pause
            exit /b 1
        )
        python -m venv "%VENV_DIR%"
    )

    if errorlevel 1 (
        echo ERROR: Could not create the Python environment.
        pause
        exit /b 1
    )
)

set "INSTALL_REQUIRED=0"
if not exist "%REQUIREMENTS_STAMP%" set "INSTALL_REQUIRED=1"
if exist "%REQUIREMENTS_STAMP%" (
    fc /b "%REQUIREMENTS%" "%REQUIREMENTS_STAMP%" >nul 2>nul
    if errorlevel 1 set "INSTALL_REQUIRED=1"
)

if "%INSTALL_REQUIRED%"=="1" (
    echo Installing dashboard dependencies...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :install_error

    "%VENV_PYTHON%" -m pip install -r "%REQUIREMENTS%"
    if errorlevel 1 goto :install_error
    copy /y "%REQUIREMENTS%" "%REQUIREMENTS_STAMP%" >nul
)

echo Opening the dashboard...
echo Keep this window open while using the dashboard.
echo.

"%VENV_PYTHON%" -m streamlit run "%~dp0app\dashboard_app.py"

echo.
echo The dashboard has stopped.
pause
exit /b 0

:install_error
echo.
echo ERROR: Dependency installation failed. Check the internet connection above.
pause
exit /b 1
