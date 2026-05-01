@echo off
setlocal
call "%~dp0_3ddfa_paths.bat"

echo [MVFS_ROOT]  %MVFS_ROOT%
echo [DDDFA_ROOT] %DDDFA_ROOT%
echo [DDDFA_VENV] %DDDFA_VENV%

where uv >nul 2>nul
if not %errorlevel%==0 (
    echo [ERROR] uv not found. Install:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

if not exist "%DDDFA_ROOT%" (
    echo [ERROR] 3DDFA-V3 root not found:
    echo   %DDDFA_ROOT%
    exit /b 1
)

uv venv "%DDDFA_VENV%" --python 3.8
if not exist "%DDDFA_VENV%\Scripts\activate.bat" exit /b 1
call "%DDDFA_VENV%\Scripts\activate.bat"

uv pip install -r "%MVFS_ROOT%\mvfs_env_setup_win_uv\env\requirements-3ddfa-py38-cu113.txt"
if not %errorlevel%==0 exit /b 1

if exist "%DDDFA_ROOT%\requirements.txt" (
    uv pip install -r "%DDDFA_ROOT%\requirements.txt"
)

python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo [DONE] 3DDFA env ready.
endlocal
