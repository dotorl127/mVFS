@echo off
setlocal

rem Usage:
rem   bat\23_teacher_train.bat <person_name> <source|target>
rem Example:
rem   bat\23_teacher_train.bat person_x source

set "PERSON=%~1"
set "ROLE=%~2"
if "%PERSON%"=="" (
    echo Usage: bat\23_teacher_train.bat ^<person_name^> ^<source^|target^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=source"

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

set "TEACHER_DIR=%MVFS_ROOT%\workspace\%PERSON%\teacher_%ROLE%"
set "INDEX=%TEACHER_DIR%\teacher_index_final.jsonl"
if not exist "%INDEX%" (
    if exist "%TEACHER_DIR%\teacher_index_with_id.jsonl" (
        set "INDEX=%TEACHER_DIR%\teacher_index_with_id.jsonl"
    ) else (
        set "INDEX=%TEACHER_DIR%\teacher_index.jsonl"
    )
)

set "OUT_DIR=%MVFS_ROOT%\checkpoints\teacher\%PERSON%_%ROLE%"

python "%MVFS_ROOT%\train\train_teacher.py" ^
  --index "%INDEX%" ^
  --output "%OUT_DIR%" ^
  --pretrained "%MVFS_PRETRAINED_SD_TURBO%" ^
  --device %MVFS_DEVICE% ^
  --fp16 ^
  --batch-size 1 ^
  --lambda-noise 1.0 ^
  --lambda-recon 0.1 ^
  --lambda-id 0.05 ^
  --fixed-high-timestep 999

endlocal
