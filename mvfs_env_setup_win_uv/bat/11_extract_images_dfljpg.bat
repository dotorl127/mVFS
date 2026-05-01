@echo off
setlocal
rem Usage: 11_extract_images_dfljpg.bat <images_dir> <out_dir>
if "%~1"=="" (
  echo Usage: 11_extract_images_dfljpg.bat ^<images_dir^> ^<out_dir^>
  exit /b 1
)
if "%~2"=="" (
  echo Usage: 11_extract_images_dfljpg.bat ^<images_dir^> ^<out_dir^>
  exit /b 1
)
call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1
python "%DDDFA_ROOT%\extract_images_dfljpg.py" extract ^
  --images-dir "%~1" ^
  --output "%~2" ^
  --device %DDDFA_DEVICE% ^
  --detector %DDDFA_DETECTOR% ^
  --backbone %DDDFA_BACKBONE% ^
  --image-size %DDDFA_IMAGE_SIZE% ^
  --face-type %DDDFA_FACE_TYPE% ^
  --recursive ^
  --skip-existing
endlocal
