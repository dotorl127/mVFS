@echo off
rem ============================================================
rem Activate 3DDFA env.
rem This file calls _3ddfa_paths.bat internally.
rem ============================================================

call "%~dp0_3ddfa_paths.bat"

if not exist "%DDDFA_VENV%\Scripts\activate.bat" (
    echo [ERROR] 3DDFA virtual env not found:
    echo   %DDDFA_VENV%
    echo Run:
    echo   bat\3ddfa_setup_env.bat
    exit /b 1
)

call "%DDDFA_VENV%\Scripts\activate.bat"

if not exist "%DDDFA_ROOT%" (
    echo [ERROR] 3DDFA-V3 directory not found:
    echo   %DDDFA_ROOT%
    echo Expected layout:
    echo   mvfs\3ddfa-v3
    exit /b 1
)

exit /b 0
