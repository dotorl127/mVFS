@echo off
setlocal

rem MVFS teacher full UNet train with late ID loss.
rem ID loss backend: facenet-pytorch InceptionResnetV1(vggface2), chosen for RTX 3060 12GB.
rem Usage:
rem   bat\26_teacher_train_late_idloss_facenet.bat <teacher_workspace> <run_name>

set "WS=%~1"
set "RUN_NAME=%~2"

if "%WS%"=="" (
    echo Usage: bat\26_teacher_train_late_idloss_facenet.bat ^<teacher_workspace^> ^<run_name^>
    exit /b 1
)
if "%RUN_NAME%"=="" set "RUN_NAME=late_idloss_facenet_run01"

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
  --grad-accum-steps 8 ^
  --num-workers 2 ^
  --epochs 100 ^
  --max-steps 6500 ^
  --lr 1e-4 ^
  --weight-decay 1e-3 ^
  --grad-clip 1.0 ^
  --fp16 ^
  --train-unet ^
  --gradient-checkpointing ^
  --id-dim 512 ^
  --num-id-tokens 4 ^
  --fixed-high-timestep 999 ^
  --noise-loss mse ^
  --lambda-noise 1.0 ^
  --lambda-recon 10.0 ^
  --lambda-id 1.0 ^
  --id-loss-start-step 5000 ^
  --id-loss-backend facenet ^
  --facenet-pretrained vggface2 ^
  --id-loss-target identity ^
  --id-loss-every 1 ^
  --recon-every 1 ^
  --log-every 10 ^
  --save-every 1000 ^
  --debug-image-every 10 ^
  --debug-max-save 200 ^
  --debug-num-samples 1

endlocal
