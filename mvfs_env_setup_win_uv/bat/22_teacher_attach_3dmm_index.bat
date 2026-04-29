@echo off
setlocal

rem Usage:
rem   bat\22_teacher_attach_3dmm_index.bat <person_name> <source|target>
rem Example:
rem   bat\22_teacher_attach_3dmm_index.bat person_x source

set "PERSON=%~1"
set "ROLE=%~2"
if "%PERSON%"=="" (
    echo Usage: bat\22_teacher_attach_3dmm_index.bat ^<person_name^> ^<source^|target^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=source"

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

set "TEACHER_DIR=%MVFS_ROOT%\workspace\%PERSON%\teacher_%ROLE%"
set "INDEX=%TEACHER_DIR%\teacher_index_with_id.jsonl"
set "OUT_INDEX=%TEACHER_DIR%\teacher_index_final.jsonl"

if not exist "%INDEX%" (
    echo [WARN] teacher_index_with_id.jsonl not found. Using teacher_index.jsonl instead.
    set "INDEX=%TEACHER_DIR%\teacher_index.jsonl"
)

python "%MVFS_ROOT%\scripts\attach_3dmm_to_teacher_index.py" ^
  --index "%INDEX%" ^
  --out-index "%OUT_INDEX%" ^
  --out-dir "%TEACHER_DIR%\3dmm"

endlocal
