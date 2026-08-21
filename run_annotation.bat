@echo off
setlocal

:: Check if exactly one argument is provided
if "%~1"=="" (
    echo Error: No argument provided.
    echo Usage: %~nx0 ^<path_to_video^>
    exit /b 1
)
if not "%~2"=="" (
    echo Error: Too many arguments.
    echo Usage: %~nx0 ^<path_to_video^>
    exit /b 1
)

:: Assign the argument to a variable
set "VIDEO_PATH=%~f1"

:: Video path is resolved above (relative to the caller); now switch to the
:: script directory so "tracking/annotation_tool.py" resolves too.
cd /d "%~dp0"

:: Run the annotation tool with the provided argument
python tracking/annotation_tool.py stark_st baseline "%VIDEO_PATH%"

endlocal