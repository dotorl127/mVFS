@echo off
setlocal
rem Usage: 10_extract_video_dfljpg.bat <video_path> <out_dir>
if "%~1"=="" (
  echo Usage: 10_extract_video_dfljpg.bat ^<video_path^> ^<out_dir^>
  exit /b 1
)
if "%~2"=="" (
  echo Usage: 10_extract_video_dfljpg.bat ^<video_path^> ^<out_dir^>
  exit /b 1
)
call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1
python "%DDDFA_ROOT%\extract_video_dfljpg.py" extract ^
  --video "%~1" ^
  --output "%~2" ^
  --device %DDDFA_DEVICE% ^
  --detector %DDDFA_DETECTOR% ^
  --backbone %DDDFA_BACKBONE% ^
  --image-size %DDDFA_IMAGE_SIZE% ^
  --face-type %DDDFA_FACE_TYPE% ^
  --frame-step 1
endlocal
