@echo off
setlocal
rem Usage: 20_update_a1a2_3ddfa.bat <dataset_root>
if "%~1"=="" (
  echo Usage: 20_update_a1a2_3ddfa.bat ^<dataset_root^>
  exit /b 1
)
call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1
python "%DDDFA_ROOT%\update_a1a2_3ddfa_metadata.py" ^
  --dataset-root "%~1" ^
  --splits A1,A2 ^
  --device %DDDFA_DEVICE% ^
  --backbone %DDDFA_BACKBONE% ^
  --write-dfljpg ^
  --write-npz ^
  --replace-top-landmarks ^
  --skip-existing
endlocal
