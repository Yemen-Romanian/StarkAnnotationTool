@echo off
setlocal

:: Resolve paths against this script, not the caller's working directory.
cd /d "%~dp0"

:: ---------------------------------------------------------------------------
:: Environment for RTX 30/40-series (compute capability 8.6 / 8.9).
::
:: The original install_pytorch17.bat builds against CUDA 10.2, which only
:: ships cubins up to sm_75 -- Ada GPUs (sm_89) cannot run it. sm_89 needs a
:: CUDA 11.8+ build, i.e. PyTorch >= 2.0, which in turn dropped Python 3.7.
::
:: PyTorch is pinned to 2.5.x on purpose: 2.6 flipped torch.load's
:: weights_only default to True, which breaks loading the STARK .pth.tar
:: checkpoints (they carry non-tensor entries alongside 'net').
:: ---------------------------------------------------------------------------

set "ENV_NAME=stark"
set "PY_VER=3.10"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"

echo ****************** Creating conda env %ENV_NAME% (python %PY_VER%) ******************
:: Skip creation if the env is already there, so this script stays re-runnable
:: (setup_env.bat calls it unconditionally).
call conda env list | findstr /R /C:"^%ENV_NAME% " >nul
if not errorlevel 1 (
    echo Environment %ENV_NAME% already exists - skipping creation.
) else (
    call conda create -n %ENV_NAME% python=%PY_VER% -y
    if errorlevel 1 goto :error
)

call conda activate %ENV_NAME%
if errorlevel 1 goto :activate_error

echo.
echo ****************** Installing pytorch 2.5.1 + torchvision 0.20.1 (cu121) ******************
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url %TORCH_INDEX%
if errorlevel 1 goto :error

echo.
echo ****************** Installing numpy ******************
:: Intentionally unpinned: opencv-python 5.x requires numpy >= 2, so a
:: "numpy<2" pin here would just be undone by the opencv step below.
:: The numpy-1-only aliases this codebase used (np.bool, np.int) have been
:: replaced, so numpy 2.x is fine. To stay on numpy 1.x instead, pin
:: opencv-python<5 as well - the two have to move together.
python -m pip install numpy
if errorlevel 1 goto :error

echo.
echo ****************** Installing timm ******************
:: timm 0.3.2 (used by install_pytorch17.bat) imports torch._six, removed in
:: torch 2.0. 0.9.x still exposes timm.models.layers, which swin_transformer.py
:: imports DropPath / to_2tuple / trunc_normal_ from.
python -m pip install timm==0.9.16
if errorlevel 1 goto :error

echo.
echo ****************** Installing yaml ******************
python -m pip install PyYAML
if errorlevel 1 goto :error

echo.
echo ****************** Installing yacs ******************
:: Required by lib/models/stark/swin_config.py, which is imported
:: unconditionally through lib/models/stark/__init__.py.
python -m pip install yacs
if errorlevel 1 goto :error

echo.
echo ****************** Installing easydict ******************
python -m pip install easydict
if errorlevel 1 goto :error

echo.
echo ****************** Installing opencv-python ******************
python -m pip install opencv-python
if errorlevel 1 goto :error

echo.
echo ****************** Installing jpeg4py ******************
:: Required even for inference: lib/train/data/image_loader.py imports it
:: at module level, and that module is on the import chain of
:: lib.test.evaluation. The pure-python wrapper imports fine without the
:: libjpeg-turbo DLL, which is only touched if a jpeg4py loader is used.
python -m pip install jpeg4py
if errorlevel 1 goto :error

echo.
echo ****************** Installing pandas ******************
python -m pip install pandas
if errorlevel 1 goto :error

echo.
echo ****************** Installing tqdm ******************
python -m pip install tqdm
if errorlevel 1 goto :error

echo.
echo ****************** Installing colorama ******************
python -m pip install colorama
if errorlevel 1 goto :error

echo.
echo ****************** Installing scipy ******************
python -m pip install scipy
if errorlevel 1 goto :error

echo.
echo ****************** Installing lmdb ******************
python -m pip install lmdb
if errorlevel 1 goto :error

echo.
echo ****************** Installing gdown ******************
:: Used by tracking/download_weights.py to fetch the pretrained checkpoints.
python -m pip install gdown
if errorlevel 1 goto :error

echo.
echo ****************** Installing tensorboard ******************
python -m pip install tensorboard
if errorlevel 1 goto :error

:: --- Optional extras: only needed for training / benchmark evaluation. ---
:: A failure here does not block the annotation tool, so just warn.
echo.
echo ****************** Installing coco toolkit (optional) ******************
python -m pip install pycocotools
if errorlevel 1 echo WARNING: pycocotools failed to install - only needed for COCO training.

echo.
echo ****************** Verifying GPU support ******************
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'); print('compute capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')"
if errorlevel 1 goto :error

echo.
echo ****************** Installation complete! ******************
echo Next: run setup_env.bat to generate the local.py path files.
endlocal
exit /b 0

:activate_error
echo.
echo ERROR: "conda activate %ENV_NAME%" failed.
echo Run "conda init cmd.exe" once, open a new terminal, then re-run this script.
endlocal
exit /b 1

:error
echo.
echo ERROR: installation failed (exit code %errorlevel%).
endlocal
exit /b 1
