@echo off
setlocal

rem Usage:
rem   bat\3ddfa_extract_source_images.bat <person_name>
rem Example:
rem   bat\3ddfa_extract_source_images.bat person_x

set "PERSON=%~1"

if "%PERSON%"=="" (
    echo Usage: bat\3ddfa_extract_source_images.bat ^<person_name^>
    exit /b 1
)

call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1

set "IMG_DIR=%MVFS_ROOT%\workspace\%PERSON%\source\raw\images"
set "OUT_DIR=%MVFS_ROOT%\workspace\%PERSON%\source\aligned_dfljpg"
set "DBG_DIR=%MVFS_ROOT%\workspace\%PERSON%\source\debug\image_extract"
set "INDEX_OUT=%MVFS_ROOT%\workspace\%PERSON%\source\meta\image_extract_index.jsonl"

echo [IMAGES] %IMG_DIR%
echo [OUTPUT] %OUT_DIR%

python "%DDDFA_ROOT%\extract_images_dfljpg.py" extract ^
  --images-dir "%IMG_DIR%" ^
  --output "%OUT_DIR%" ^
  --device %DDDFA_DEVICE% ^
  --detector %DDDFA_DETECTOR% ^
  --backbone %DDDFA_BACKBONE% ^
  --image-size %DDDFA_IMAGE_SIZE% ^
  --face-type %DDDFA_FACE_TYPE% ^
  --recursive ^
  --skip-existing ^
  --debug-dir "%DBG_DIR%" ^
  --index-out "%INDEX_OUT%"

endlocal
