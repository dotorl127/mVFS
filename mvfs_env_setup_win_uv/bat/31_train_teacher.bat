@echo off
setlocal
rem Usage: 31_train_teacher.bat <teacher_workspace> <run_name>
if "%~1"=="" (
  echo Usage: 31_train_teacher.bat ^<teacher_workspace^> ^<run_name^>
  exit /b 1
)
set "WS=%~1"
set "RUN_NAME=%~2"
if "%RUN_NAME%"=="" set "RUN_NAME=teacher_run01"

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

set "INDEX=%WS%\meta\teacher_index.jsonl"
set "OUTDIR=%WS%\runs\%RUN_NAME%"

python "%MVFS_ROOT%\train\train_teacher.py" ^
  --index "%INDEX%" ^
  --output "%OUTDIR%" ^
  --pretrained "%MVFS_PRETRAINED_SD_TURBO%" ^
  --device %MVFS_DEVICE% ^
  --image-size 512 ^
  --batch-size 1 ^
  --grad-accum-steps 8 ^
  --num-workers 2 ^
  --epochs 100 ^
  --max-steps 65000 ^
  --lr 1e-4 ^
  --weight-decay 1e-3 ^
  --grad-clip 1.0 ^
  --fp16 ^
  --train-unet ^
  --gradient-checkpointing ^
  --condition-channels 4 ^
  --landmark-sigma 2.0 ^
  --blur-downsample-size 32 ^
  --blur-gaussian-radius 0.0 ^
  --blur-feather-sigma 0.0 ^
  --id-dim 512 ^
  --num-id-tokens 4 ^
  --fixed-high-timestep 999 ^
  --noise-loss mse ^
  --lambda-noise 1.0 ^
  --lambda-l1 10.0 ^
  --lambda-lpips 1.0 ^
  --lambda-id 1.0 ^
  --id-loss-start-step 50000 ^
  --id-loss-backend facenet ^
  --facenet-pretrained vggface2 ^
  --id-loss-target identity ^
  --lpips-net alex ^
  --log-every 10 ^
  --save-every 100 ^
  --save-every-micro 0 ^
  --debug-image-every 10 ^
  --debug-max-save 1000 ^
  --debug-num-samples 1
endlocal
