@echo off
setlocal
rem Usage: 22_update_a1a2_gaze.bat <dataset_root>
if "%~1"=="" (
  echo Usage: 22_update_a1a2_gaze.bat ^<dataset_root^>
  exit /b 1
)
.venv-gaze\Scripts\python.exe mvfs_env_setup_win_uv\tools\update_a1a2_gaze_mediapipe.py ^
  --dataset-root D:\MVFS\dataset\VGGFace2-HQ-split-balanced ^
  --pad-ratio 0.25 ^
  --min-det-conf 0.5 ^
  --debug-dir D:\MVFS\debug_gaze
endlocal
