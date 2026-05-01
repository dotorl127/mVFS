@echo off
rem Common 3DDFA paths. CALL this file from other bat files.

set "MVFS_ROOT=%~dp0..\.."
for %%I in ("%MVFS_ROOT%") do set "MVFS_ROOT=%%~fI"

if "%DDDFA_ROOT%"=="" set "DDDFA_ROOT=%MVFS_ROOT%\3DDFA-V3"
for %%I in ("%DDDFA_ROOT%") do set "DDDFA_ROOT=%%~fI"

rem 3DDFA venv must live under 3DDFA-V3 root.
if "%DDDFA_VENV%"=="" set "DDDFA_VENV=%DDDFA_ROOT%\.venv-3ddfa"
for %%I in ("%DDDFA_VENV%") do set "DDDFA_VENV=%%~fI"

if "%DDDFA_DEVICE%"=="" set "DDDFA_DEVICE=cuda"
if "%DDDFA_DETECTOR%"=="" set "DDDFA_DETECTOR=retinaface"
if "%DDDFA_BACKBONE%"=="" set "DDDFA_BACKBONE=resnet50"
if "%DDDFA_IMAGE_SIZE%"=="" set "DDDFA_IMAGE_SIZE=512"
if "%DDDFA_FACE_TYPE%"=="" set "DDDFA_FACE_TYPE=whole_face"
