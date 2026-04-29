@echo off
setlocal

rem Usage:
rem   bat\21_teacher_build_id_embeddings.bat <person_name> <source|target>
rem Example:
rem   bat\21_teacher_build_id_embeddings.bat person_x source

set "PERSON=%~1"
set "ROLE=%~2"
if "%PERSON%"=="" (
    echo Usage: bat\21_teacher_build_id_embeddings.bat ^<person_name^> ^<source^|target^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=source"

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

set "TEACHER_DIR=%MVFS_ROOT%\workspace\%PERSON%\teacher_%ROLE%"
set "INDEX=%TEACHER_DIR%\teacher_index.jsonl"
set "OUT_INDEX=%TEACHER_DIR%\teacher_index_with_id.jsonl"
set "EMB_DIR=%TEACHER_DIR%\id_embeddings"

python "%MVFS_ROOT%\scripts\build_teacher_id_embeddings.py" ^
  --index "%INDEX%" ^
  --out-index "%OUT_INDEX%" ^
  --embed-dir "%EMB_DIR%" ^
  --mean-out "%TEACHER_DIR%\id_embedding_mean.npz" ^
  --ctx-id 0 ^
  --fallback-zero

endlocal
