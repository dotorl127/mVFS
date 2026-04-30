@echo off
setlocal

call "%~dp0_mvfs_activate.bat"
if not %errorlevel%==0 exit /b 1

rem Avoid letting facenet-pytorch replace the installed CUDA torch stack.
uv pip install facenet-pytorch --no-deps

python -c "from facenet_pytorch import InceptionResnetV1; m=InceptionResnetV1(pretrained='vggface2').eval(); print('facenet-pytorch ID loss model ready')"

endlocal
