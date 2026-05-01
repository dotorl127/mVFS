@echo off
setlocal
rem Usage: 30_build_teacher_index.bat <dataset_root> <teacher_workspace>
if "%~1"=="" (
  echo Usage: 30_build_teacher_index.bat ^<dataset_root^> ^<teacher_workspace^>
  exit /b 1
)
if "%~2"=="" (
  echo Usage: 30_build_teacher_index.bat ^<dataset_root^> ^<teacher_workspace^>
  exit /b 1
)
call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1
python "%MVFS_ROOT%\scripts\build_teacher_index_a1a2.py" ^
  --dataset-root "%~1" ^
  --out-root "%~2" ^
  --fallback-zero-id
endlocal
