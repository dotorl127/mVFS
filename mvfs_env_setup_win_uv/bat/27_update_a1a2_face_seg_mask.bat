
@echo off
setlocal
if "%~1"=="" (
  echo Usage: 27_update_a1a2_face_seg_mask.bat ^<dataset_root^>
  exit /b 1
)
call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1
python "%MVFS_ROOT%\tools\update_a1a2_face_seg_mask.py" --dataset-root "%~1" --device cuda --skip-existing
endlocal
