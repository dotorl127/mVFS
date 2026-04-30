@echo off
setlocal

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

call "%~dp0_mvfs_echo_config.bat"

python "%MVFS_ROOT%\tools\download_mvfs_pretrained.py" ^
  --pretrained-root "%MVFS_ROOT%\pretrained" ^
  --sd-turbo ^
  --insightface-buffalo-l

endlocal
