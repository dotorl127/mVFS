@echo off
setlocal

call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1

call "%~dp0_3ddfa_echo_config.bat"

python "%MVFS_ROOT%\tools\download_3ddfa_assets.py" ^
  --repo-id Zidu-Wang/3DDFA-V3 ^
  --out-dir "%DDDFA_ROOT%\assets"

endlocal
