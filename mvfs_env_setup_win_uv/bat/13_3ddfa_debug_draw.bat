@echo off
setlocal

rem Usage:
rem   bat\3ddfa_debug_draw.bat <person_name> <target|source>
rem Example:
rem   bat\3ddfa_debug_draw.bat person_x target

set "PERSON=%~1"
set "ROLE=%~2"

if "%PERSON%"=="" (
    echo Usage: bat\3ddfa_debug_draw.bat ^<person_name^> ^<target^|source^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=target"

call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1

set "IN_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\aligned_dfljpg"
set "OUT_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\debug\aligned_landmarks"

if exist "%DDDFA_ROOT%\extract_video_dfljpg.py" (
    python "%DDDFA_ROOT%\extract_video_dfljpg.py" debug-draw ^
      --input "%IN_DIR%" ^
      --output "%OUT_DIR%"
) else (
    echo [ERROR] extract_video_dfljpg.py not found under:
    echo   %DDDFA_ROOT%
    exit /b 1
)

endlocal
