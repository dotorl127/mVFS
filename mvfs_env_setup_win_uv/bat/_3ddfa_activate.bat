@echo off
call "%~dp0_3ddfa_paths.bat"

if not exist "%DDDFA_ROOT%" (
    echo [ERROR] 3DDFA-V3 directory not found:
    echo   %DDDFA_ROOT%
    echo Expected:
    echo   D:\MVFS\3DDFA-V3
    exit /b 1
)

if not exist "%DDDFA_VENV%\Scripts\activate.bat" (
    echo [ERROR] 3DDFA virtual env not found:
    echo   %DDDFA_VENV%
    echo Run:
    echo   mvfs_env_setup_win_uv\bat\01_setup_3ddfa_env.bat
    exit /b 1
)

call "%DDDFA_VENV%\Scripts\activate.bat"
exit /b 0
