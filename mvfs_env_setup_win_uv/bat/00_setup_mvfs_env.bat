@echo off
setlocal
call "%~dp0_mvfs_paths.bat"

echo [MVFS_ROOT] %MVFS_ROOT%
echo [MVFS_VENV] %MVFS_VENV%

where uv >nul 2>nul
if not %errorlevel%==0 (
    echo [ERROR] uv not found. Install:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

uv venv "%MVFS_VENV%" --python 3.10
if not exist "%MVFS_VENV%\Scripts\activate.bat" exit /b 1
call "%MVFS_VENV%\Scripts\activate.bat"

uv pip install -r "%MVFS_ROOT%\mvfs_env_setup_win_uv\env\requirements-mvfs-cu121.txt"
if not %errorlevel%==0 exit /b 1

rem facenet-pytorch is for late ID loss. Install without deps to avoid torch downgrade.
uv pip install facenet-pytorch --no-deps

python "%MVFS_ROOT%\tools\check_mvfs_gpu_env.py"

echo [DONE] MVFS env ready.
endlocal
