@echo off
setlocal
rem Usage: 12_manual_extract_frames_dfljpg.bat <frames_dir> <out_dir>
if "%~1"=="" (
  echo Usage: 12_manual_extract_frames_dfljpg.bat ^<frames_dir^> ^<out_dir^>
  exit /b 1
)
if "%~2"=="" (
  echo Usage: 12_manual_extract_frames_dfljpg.bat ^<frames_dir^> ^<out_dir^>
  exit /b 1
)
call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1
python "%DDDFA_ROOT%\manual_extract_frames_dfljpg.py" ^
  --frames-dir "%~1" ^
  --output "%~2" ^
  --device %DDDFA_DEVICE% ^
  --detector %DDDFA_DETECTOR% ^
  --backbone %DDDFA_BACKBONE% ^
  --image-size %DDDFA_IMAGE_SIZE% ^
  --face-type %DDDFA_FACE_TYPE%
endlocal
