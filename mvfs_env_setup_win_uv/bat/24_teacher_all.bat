@echo off
setlocal

rem Full teacher blur reconstruction pipeline:
rem   1. build blur+landmark condition
rem   2. build InsightFace ID embeddings
rem   3. attach 3DMM metadata if DFLJPG has it
rem   4. train teacher
rem
rem Usage:
rem   bat\24_teacher_all.bat <person_name> <source|target>

set "PERSON=%~1"
set "ROLE=%~2"
if "%PERSON%"=="" (
    echo Usage: bat\24_teacher_all.bat ^<person_name^> ^<source^|target^>
    exit /b 1
)
if "%ROLE%"=="" set "ROLE=source"

call "%~dp020_teacher_build_blur_conditions.bat" "%PERSON%" "%ROLE%"
if not %errorlevel%==0 exit /b 1

call "%~dp021_teacher_build_id_embeddings.bat" "%PERSON%" "%ROLE%"
if not %errorlevel%==0 exit /b 1

call "%~dp022_teacher_attach_3dmm_index.bat" "%PERSON%" "%ROLE%"
if not %errorlevel%==0 exit /b 1

call "%~dp023_teacher_train.bat" "%PERSON%" "%ROLE%"
if not %errorlevel%==0 exit /b 1

endlocal
