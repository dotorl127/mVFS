@echo off
setlocal
rem Usage: 40_preprocess_a1a2_all.bat <dataset_root> <teacher_workspace>
if "%~1"=="" (
  echo Usage: 40_preprocess_a1a2_all.bat ^<dataset_root^> ^<teacher_workspace^>
  exit /b 1
)
if "%~2"=="" (
  echo Usage: 40_preprocess_a1a2_all.bat ^<dataset_root^> ^<teacher_workspace^>
  exit /b 1
)
call "%~dp020_update_a1a2_3ddfa.bat" "%~1"
if not %errorlevel%==0 exit /b 1
call "%~dp021_update_a1a2_faceseg.bat" "%~1"
if not %errorlevel%==0 exit /b 1
call "%~dp030_build_teacher_index.bat" "%~1" "%~2"
if not %errorlevel%==0 exit /b 1
endlocal
