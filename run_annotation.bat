@echo off
setlocal

:: Check that a video path is provided
if "%~1"=="" (
    echo Error: No argument provided.
    echo Usage: %~nx0 ^<path_to_video^> [extra options, e.g. --device cpu]
    exit /b 1
)

:: Assign the first argument to a variable; anything after it is forwarded
:: to the annotation tool (e.g. "--device cpu" to run without a GPU).
set "VIDEO_PATH=%~f1"
shift
set "EXTRA_ARGS="
:collect_args
if "%~1"=="" goto args_done
set "EXTRA_ARGS=%EXTRA_ARGS% %1"
shift
goto collect_args
:args_done

:: Video path is resolved above (relative to the caller); now switch to the
:: script directory so "tracking/annotation_tool.py" resolves too.
cd /d "%~dp0"

:: Run the annotation tool with the provided argument
python tracking/annotation_tool.py stark_st baseline "%VIDEO_PATH%"%EXTRA_ARGS%

endlocal