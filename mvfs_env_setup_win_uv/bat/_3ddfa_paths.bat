@echo off
rem ============================================================
rem MVFS / 3DDFA-V3 common paths
rem This file is meant to be CALLed by other bat files.
rem ============================================================

rem bat\_3ddfa_paths.bat -> mvfs root is one level above bat
set "MVFS_ROOT=%~dp0..\.."
for %%I in ("%MVFS_ROOT%") do set "MVFS_ROOT=%%~fI"

rem Default 3DDFA-V3 directory. Override by setting DDDFA_ROOT before call.
if "%DDDFA_ROOT%"=="" set "DDDFA_ROOT=%MVFS_ROOT%\3DDFA-V3"
for %%I in ("%DDDFA_ROOT%") do set "DDDFA_ROOT=%%~fI"

rem Default venv path. Override by setting DDDFA_VENV before call.
if "%DDDFA_VENV%"=="" set "DDDFA_VENV=%MVFS_ROOT%\.venv-3ddfa"
for %%I in ("%DDDFA_VENV%") do set "DDDFA_VENV=%%~fI"

rem Default device / detector / backbone
if "%DDDFA_DEVICE%"=="" set "DDDFA_DEVICE=cuda"
if "%DDDFA_DETECTOR%"=="" set "DDDFA_DETECTOR=retinaface"
if "%DDDFA_BACKBONE%"=="" set "DDDFA_BACKBONE=resnet50"
if "%DDDFA_IMAGE_SIZE%"=="" set "DDDFA_IMAGE_SIZE=512"
if "%DDDFA_FACE_TYPE%"=="" set "DDDFA_FACE_TYPE=whole_face"
