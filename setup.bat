@echo off
echo ===================================================
echo   Cloud Masking Suite - Setup Script
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH.
    echo Please install Python (3.8+) and try again.
    pause
    exit /b
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [1/3] Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
) else (
    echo [1/3] Virtual environment (venv) already exists. Skipping creation.
)

:: Install requirements
echo [2/3] Installing dependencies from requirements.txt...
venv\Scripts\python.exe -m pip install --upgrade pip >nul
venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

:: Set up VS Code settings so it automatically picks up the virtual environment
echo [3/3] Setting up VS Code workspace configuration...
if not exist ".vscode" (
    mkdir .vscode
)

(
echo {
echo     "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"
echo }
) > .vscode\settings.json

echo.
echo ===================================================
echo   Setup Complete!
echo ===================================================
echo  1. Open this folder in VS Code. It will automatically
echo     pick up the python environment.
echo  2. Run 'python gui.py' to launch the GUI.
echo ===================================================
echo.
pause
