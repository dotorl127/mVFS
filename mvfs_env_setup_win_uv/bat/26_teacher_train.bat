@echo off
setlocal

rem Usage:
rem   bat\26_teacher_train.bat <teacher_workspace> <run_name>

set "WS=%~1"
set "RUN_NAME=%~2"

if "%WS%"=="" (
    echo Usage: bat\26_teacher_train.bat ^<teacher_workspace^> ^<run_name^>
    exit /b 1
)
if "%RUN_NAME%"=="" (
    echo Usage: bat\26_teacher_train.bat ^<teacher_workspace^> ^<run_name^>
    exit /b 1
)

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

set "INDEX=%WS%\meta\teacher_index.jsonl"
set "OUTDIR=%WS%\runs\%RUN_NAME%"

python "%MVFS_ROOT%\train\train_teacher.py" ^
  --index "%INDEX%" ^
  --output "%OUTDIR%" ^
  --pretrained "stabilityai/sd-turbo" ^
  --device "cuda" ^
  --image-size 512 ^
  --batch-size 1 ^
  --num-workers 2 ^
  --epochs 1 ^
  --max-steps 10000 ^
  --lr 1e-4 ^
  --weight-decay 1e-2 ^
  --grad-clip 1.0 ^
  --fp16 ^
  --train-unet ^
  --id-dim 512 ^
  --num-id-tokens 4 ^
  --fixed-high-timestep 999 ^
  --noise-loss mse ^
  --lambda-noise 1.0 ^
  --lambda-recon 0.1 ^
  --recon-every 4 ^
  --log-every 10 ^
  --save-every 1000 ^
  --debug-image-every 10 ^
  --debug-max-save 200 ^
  --debug-num-samples 1

endlocal
