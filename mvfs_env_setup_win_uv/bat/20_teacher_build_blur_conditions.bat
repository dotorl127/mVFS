@echo off
setlocal

rem Usage:
rem   bat\20_teacher_build_blur_conditions.bat <person_name> <source|target>
rem Example:
rem   bat\20_teacher_build_blur_conditions.bat person_x source

set "PERSON=%~1"
set "ROLE=%~2"
if "%PERSON%"=="" (
    echo Usage: bat\20_teacher_build_blur_conditions.bat ^<person_name^> ^<source^|target^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=source"

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

set "ALIGNED_DIR=%MVFS_ROOT%\workspace\%PERSON%\%ROLE%\aligned_dfljpg"
set "TEACHER_DIR=%MVFS_ROOT%\workspace\%PERSON%\teacher_%ROLE%"

python "%MVFS_ROOT%\scripts\build_teacher_blur_conditions.py" ^
  --aligned-dir "%ALIGNED_DIR%" ^
  --out-dir "%TEACHER_DIR%" ^
  --downsample 8 ^
  --expand 0.18 ^
  --debug

endlocal
