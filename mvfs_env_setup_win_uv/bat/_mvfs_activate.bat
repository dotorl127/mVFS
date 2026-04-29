@echo off
call "%~dp0_mvfs_paths.bat"

if not exist "%MVFS_VENV%\Scripts\activate.bat" (
    echo [ERROR] MVFS virtual env not found:
    echo   %MVFS_VENV%
    echo Run MVFS env setup first.
    exit /b 1
)

call "%MVFS_VENV%\Scripts\activate.bat"
exit /b 0
