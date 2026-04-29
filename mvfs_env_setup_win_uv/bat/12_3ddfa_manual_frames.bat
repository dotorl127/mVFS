@echo off
setlocal

rem Usage:
rem   bat\3ddfa_manual_frames.bat <person_name> <target|source>
rem Example:
rem   bat\3ddfa_manual_frames.bat person_x target

set "PERSON=%~1"
set "ROLE=%~2"

if "%PERSON%"=="" (
    echo Usage: bat\3ddfa_manual_frames.bat ^<person_name^> ^<target^|source^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=target"

call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1

set "FRAMES_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\frames"
set "OUT_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\aligned_dfljpg"
set "DBG_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\debug\manual_landmarks"

echo [FRAMES] %FRAMES_DIR%
echo [OUTPUT] %OUT_DIR%

python "%DDDFA_ROOT%\manual_extract_frames_dfljpg.py" ^
  --frames-dir "%FRAMES_DIR%" ^
  --output "%OUT_DIR%" ^
  --device %DDDFA_DEVICE% ^
  --detector %DDDFA_DETECTOR% ^
  --backbone %DDDFA_BACKBONE% ^
  --image-size %DDDFA_IMAGE_SIZE% ^
  --face-type %DDDFA_FACE_TYPE% ^
  --debug-dir "%DBG_DIR%" ^
  --continuous-skip-delay-ms 60

endlocal
