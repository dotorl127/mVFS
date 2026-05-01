@echo off
call "%~dp0_mvfs_paths.bat"

if not exist "%MVFS_VENV%\Scripts\activate.bat" (
    echo [ERROR] MVFS virtual env not found:
    echo   %MVFS_VENV%
    echo Run:
    echo   mvfs_env_setup_win_uv\bat\00_setup_mvfs_env.bat
    exit /b 1
)

call "%MVFS_VENV%\Scripts\activate.bat"
exit /b 0
