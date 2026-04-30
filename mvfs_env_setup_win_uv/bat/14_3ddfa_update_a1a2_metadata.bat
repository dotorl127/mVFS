@echo off
setlocal
rem Update existing id/A1,A2 aligned DFLJPG dataset with 3DDFA-V3 metadata.
rem V2: no face detector, no face_box, no mtcnn, no renderer forward.
rem Usage:
rem   bat\14_3ddfa_update_a1a2_metadata.bat <dataset_root>

set "DATASET_ROOT=%~1"
if "%DATASET_ROOT%"=="" (
    echo Usage: bat\14_3ddfa_update_a1a2_metadata.bat ^<dataset_root^>
    exit /b 1
)

call "%~dp0_3ddfa_activate.bat"
if not %errorlevel%==0 exit /b 1

python "%DDDFA_ROOT%\update_a1a2_3ddfa_metadata_aligned_v3.py" ^
  --dataset-root "%DATASET_ROOT%" ^
  --splits A1,A2 ^
  --device %DDDFA_DEVICE% ^
  --backbone %DDDFA_BACKBONE% ^
  --write-dfljpg ^
  --write-npz ^
  --replace-top-landmarks ^
  --skip-existing

endlocal
