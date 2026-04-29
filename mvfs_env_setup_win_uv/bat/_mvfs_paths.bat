@echo off
rem Common MVFS paths. This file is meant to be CALLed.

set "MVFS_ROOT=%~dp0.."
for %%I in ("%MVFS_ROOT%") do set "MVFS_ROOT=%%~fI"

if "%MVFS_VENV%"=="" set "MVFS_VENV=%MVFS_ROOT%\.venv-mvfs"
for %%I in ("%MVFS_VENV%") do set "MVFS_VENV=%%~fI"

if "%MVFS_DEVICE%"=="" set "MVFS_DEVICE=cuda"
if "%MVFS_PRETRAINED_SD_TURBO%"=="" set "MVFS_PRETRAINED_SD_TURBO=%MVFS_ROOT%\pretrained\sd-turbo"
