@echo off
setlocal

rem Usage:
rem   bat\3ddfa_extract_video.bat <person_name> <target|source> <video_filename>
rem Example:
rem   bat\3ddfa_extract_video.bat person_x target target.mp4
rem   bat\3ddfa_extract_video.bat person_x source source.mp4

set "PERSON=%~1"
set "ROLE=%~2"
set "VIDEO_NAME=%~3"

if "%PERSON%"=="" (
    echo Usage: bat\3ddfa_extract_video.bat ^<person_name^> ^<target^|source^> ^<video_filename^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=target"
if "%VIDEO_NAME%"=="" set "VIDEO_NAME=target.mp4"

call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1

set "VIDEO_PATH=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\raw\videos\%VIDEO_NAME%"
set "OUT_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\aligned_dfljpg"
set "DBG_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\debug\auto_landmarks"

echo [VIDEO]  %VIDEO_PATH%
echo [OUTPUT] %OUT_DIR%

python "%DDDFA_ROOT%\extract_video_dfljpg.py" extract ^
  --video "%VIDEO_PATH%" ^
  --output "%OUT_DIR%" ^
  --device %DDDFA_DEVICE% ^
  --detector %DDDFA_DETECTOR% ^
  --backbone %DDDFA_BACKBONE% ^
  --image-size %DDDFA_IMAGE_SIZE% ^
  --face-type %DDDFA_FACE_TYPE% ^
  --frame-step 1 ^
  --debug-dir "%DBG_DIR%"

endlocal
