@echo off
setlocal

:: Resolve paths against this script, not the caller's working directory.
cd /d "%~dp0"

:: ---------------------------------------------------------------------------
:: Creates the conda env, generates the local.py path files and fetches weights.
:: (python 3.10 / pytorch 2.5.1 + cu121, see install_pytorch25.bat).
::
:: install_pytorch25.bat below creates the env; comment it out to only
:: regenerate local.py and re-check the weights.
:: ---------------------------------------------------------------------------

set "ENV_NAME=stark"

:: Explicit path: with NoDefaultCurrentDirectoryInExePath set, cmd will not
:: resolve a bare script name from the current directory.
call "%~dp0install_pytorch25.bat"
if errorlevel 1 goto :install_error

echo ****************** Generating local.py path files ******************
call conda run --no-capture-output -n %ENV_NAME% python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
if errorlevel 1 goto :error

echo.
echo ****************** Downloading STARK-ST50 weights ******************
:: Skips the download if the checkpoint is already there.
:: For STARK-ST101 instead:  download_weights.bat baseline_R101
call conda run --no-capture-output -n %ENV_NAME% python tracking/download_weights.py --variant baseline
if errorlevel 1 goto :weights_error
echo.
echo ****************** Setup complete! ******************
echo Annotate a video with:
echo   conda activate %ENV_NAME%
echo   run_annotation.bat ^<path_to_video^>
endlocal
exit /b 0

:weights_error
echo.
echo ERROR: weight download failed.
echo Everything else is set up - only the checkpoint is missing.
echo Retry with:  download_weights.bat
endlocal
exit /b 1

:install_error
echo.
echo ERROR: install_pytorch25.bat failed or could not be found.
endlocal
exit /b 1

:error
echo.
echo ERROR: setup failed (exit code %errorlevel%).
echo Is the "%ENV_NAME%" conda env created? Run install_pytorch25.bat first.
endlocal
exit /b 1
