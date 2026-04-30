@echo off
setlocal
rem Usage:
rem   bat\25_teacher_build_from_a1a2.bat <dataset_root> <out_root>

set "DATASET_ROOT=%~1"
set "OUT_ROOT=%~2"

if "%DATASET_ROOT%"=="" (
    echo Usage: bat\25_teacher_build_from_a1a2.bat ^<dataset_root^> ^<out_root^>
    exit /b 1
)
if "%OUT_ROOT%"=="" (
    echo Usage: bat\25_teacher_build_from_a1a2.bat ^<dataset_root^> ^<out_root^>
    exit /b 1
)

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

python "%MVFS_ROOT%\scripts\build_teacher_dataset_from_a1a2.py" ^
  --dataset-root "%DATASET_ROOT%" ^
  --out-root "%OUT_ROOT%" ^
  --a1-name A1 ^
  --a2-name A2 ^
  --id-source A1 ^
  --id-embed-mode direct ^
  --landmark-source 3ddfa ^
  --downsample 32 ^
  --expand 0.20 ^
  --blur-sigma 2.0 ^
  --feather-sigma 8.0 ^
  --debug ^
  --fallback-zero-id

endlocal
