@echo off
setlocal enabledelayedexpansion

call "%~dp0_mvfs_paths.bat"

echo [MVFS] setup env: Python 3.10 + torch cu121
call "%~dp0_mvfs_echo_config.bat"

where uv >nul 2>nul
if not %errorlevel%==0 (
    echo [ERROR] uv not found.
    echo Run:
    echo   bat\00_install_uv_win.bat
    exit /b 1
)

uv venv "%MVFS_VENV%" --python 3.10
if not exist "%MVFS_VENV%\Scripts\activate.bat" (
    echo [ERROR] Failed to create venv:
    echo   %MVFS_VENV%
    exit /b 1
)

call "%MVFS_VENV%\Scripts\activate.bat"

echo [MVFS] Installing requirements...
uv pip install -r "%MVFS_ROOT%\env\requirements-mvfs-cu121.txt"
if not %errorlevel%==0 (
    echo [ERROR] requirement install failed.
    exit /b 1
)

echo [MVFS] GPU check...
python "%MVFS_ROOT%\tools\check_mvfs_gpu_env.py"
if not %errorlevel%==0 (
    echo [WARN] GPU check failed. Read messages above.
)

echo.
echo [DONE] MVFS env ready.
echo Activate:
echo   call "%MVFS_VENV%\Scripts\activate.bat"
endlocal
