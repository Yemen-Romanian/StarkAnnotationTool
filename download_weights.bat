@echo off
setlocal

:: Resolve paths against this script, not the caller's working directory.
cd /d "%~dp0"

:: ---------------------------------------------------------------------------
:: Download pretrained STARK-ST weights into
::   checkpoints\train\stark_st2\<variant>\STARKST_ep0050.pth.tar
::
:: Usage:
::   download_weights.bat                       :: STARK-ST50 (baseline)
::   download_weights.bat baseline_R101         :: STARK-ST101
::   download_weights.bat --all
::   download_weights.bat baseline --force
:: ---------------------------------------------------------------------------

set "ENV_NAME=stark"

if "%~1"=="" (
    set "ARGS=--variant baseline"
) else if "%~1"=="--all" (
    set "ARGS=--all %~2"
) else (
    set "ARGS=--variant %~1 %~2"
)

echo ****************** Downloading STARK-ST weights ******************
call conda run --no-capture-output -n %ENV_NAME% python tracking/download_weights.py %ARGS%
if errorlevel 1 goto :error

endlocal
exit /b 0

:error
echo.
echo ERROR: weight download failed (exit code %errorlevel%).
echo Google Drive rate-limits anonymous downloads; retrying later usually works.
echo You can also grab the file by hand from the MODEL_ZOO.md "model" link and save it as
echo   checkpoints\train\stark_st2\baseline\STARKST_ep0050.pth.tar
endlocal
exit /b 1
