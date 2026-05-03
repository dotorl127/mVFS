@echo off
setlocal
rem Usage: 21_update_a1a2_faceseg.bat <dataset_root>
if "%~1"=="" (
  echo Usage: 21_update_a1a2_faceseg.bat ^<dataset_root^>
  exit /b 1
)
call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1
python "%MVFS_ROOT%\mvfs_env_setup_win_uv\tools\update_a1a2_face_seg_mask.py" ^
  --dataset-root "%~1" ^
  --device %MVFS_DEVICE% ^
  --skip-existing
endlocal
