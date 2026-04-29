@echo off
setlocal enabledelayedexpansion

call "%~dp0_3ddfa_paths.bat"

echo [3DDFA-V3] setup official-like env: Python 3.8 + torch cu102
call "%~dp0_3ddfa_echo_config.bat"

where uv >nul 2>nul
if not %errorlevel%==0 (
    echo [ERROR] uv not found.
    echo Install uv first:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

uv venv "%DDDFA_VENV%" --python 3.8
if not exist "%DDDFA_VENV%\Scripts\activate.bat" (
    echo [ERROR] Failed to create venv:
    echo   %DDDFA_VENV%
    exit /b 1
)

call "%DDDFA_VENV%\Scripts\activate.bat"

echo [3DDFA-V3] Installing official-like requirements...
uv pip install -r "%MVFS_ROOT%\env\requirements-3ddfa-official-py38-cu102-win.txt"
if not %errorlevel%==0 (
    echo [ERROR] requirement install failed.
    exit /b 1
)

if exist "%DDDFA_ROOT%\requirements.txt" (
    echo [3DDFA-V3] Installing repo requirements.txt...
    uv pip install -r "%DDDFA_ROOT%\requirements.txt"
) else (
    echo [INFO] %DDDFA_ROOT%\requirements.txt not found. Skipping.
)

echo [3DDFA-V3] GPU check...
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo [DONE] 3DDFA-V3 env ready.
endlocal
