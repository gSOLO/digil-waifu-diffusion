:: Digil Diffusion
::
:: Check model at /models/ldm/stable-diffusion-v1/model.ckpt
::
::	model.ckpt
::	"models\ldm\stable-diffusion-v1\model.ckpt"
::
::	Stable Diffusion model not found: you need to place model.ckpt file into same directory as this file.
::	Stable Diffusion model not found.
::
:: Check GFPGAN model at /GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth
::
::	GFPGANv1.3.pth
::	"GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth"
::
::	GFPGAN not found: you need to place GFPGANv1.3.pth file into same directory as this file.
::	GFPGAN not found.
::
:: Activate Conda
call C:\ProgramData\Miniconda3\Scripts\activate.bat	 

 
@echo off

if not defined PYTHON (set PYTHON=python)
if not defined GIT (set GIT=git)
if not defined COMMANDLINE_ARGS (set COMMANDLINE_ARGS=%*)
if not defined VENV_DIR (set VENV_DIR=venv)
if not defined TORCH_COMMAND (set TORCH_COMMAND=pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113)
if not defined REQS_FILE (set REQS_FILE=requirements_versions.txt)

mkdir tmp 2>NUL

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_git
echo Couldn't launch python
goto :show_stdout_stderr

:check_git
%GIT% --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :setup_venv
echo Couldn't launch git
goto :show_stdout_stderr

:setup_venv
if [%VENV_DIR%] == [-] goto :skip_venv

dir %VENV_DIR%\Scripts\Python.exe >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv %VENV_DIR% >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv
echo Unable to create venv in directory %VENV_DIR%
goto :show_stdout_stderr

:activate_venv
set PYTHON="%~dp0%VENV_DIR%\Scripts\Python.exe"
%PYTHON% --version
echo venv %PYTHON%
goto :install_torch

:skip_venv
%PYTHON% --version

:install_torch

%PYTHON% -c "import torch" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_gpu
echo Installing torch...
%PYTHON% -m %TORCH_COMMAND% >tmp/stdout.txt 2>tmp/stderr.txt

if %ERRORLEVEL% == 0 goto :check_gpu
echo Failed to install torch
goto :show_stdout_stderr

:check_gpu
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_sd_reqs
echo Torch is not able to use GPU
goto :show_stdout_stderr

:install_sd_reqs
%PYTHON% -c "import transformers; import wheel" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_k_diff
echo Installing SD requirements...
%PYTHON% -m pip install wheel transformers==4.19.2 diffusers invisible-watermark --prefer-binary >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_k_diff
goto :show_stdout_stderr

:install_k_diff
%PYTHON% -c "import k_diffusion.sampling" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_GFPGAN
echo Installing K-Diffusion...
%PYTHON% -m pip install git+https://github.com/crowsonkb/k-diffusion.git@1a0703dfb7d24d8806267c3e7ccc4caf67fd1331 --prefer-binary --only-binary=psutil >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_GFPGAN
goto :show_stdout_stderr


:install_GFPGAN
%PYTHON% -c "import gfpgan" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_reqs
echo Installing GFPGAN
%PYTHON% -m pip install git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379 --prefer-binary >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_reqs
goto :show_stdout_stderr

:install_reqs
%PYTHON% -c "import omegaconf; import fonts; import timm" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :make_dirs
echo Installing requirements...
%PYTHON% -m pip install -r %REQS_FILE% --prefer-binary >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :make_dirs
goto :show_stdout_stderr

:make_dirs
mkdir repositories 2>NUL

if exist repositories\stable-diffusion goto :clone_transformers
echo Cloning Stable Difusion repository...
%GIT% clone https://github.com/CompVis/stable-diffusion.git repositories\stable-diffusion >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :clone_transformers
goto :show_stdout_stderr

:clone_transformers
if exist repositories\taming-transformers goto :clone_codeformer
echo Cloning Taming Transforming repository...
%GIT% clone https://github.com/CompVis/taming-transformers.git repositories\taming-transformers >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :clone_codeformer
goto :show_stdout_stderr

:clone_codeformer
if exist repositories\CodeFormer goto :install_codeformer_reqs
echo Cloning CodeFormer repository...
%GIT% clone https://github.com/sczhou/CodeFormer.git repositories\CodeFormer >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_codeformer_reqs
goto :show_stdout_stderr

:install_codeformer_reqs
%PYTHON% -c "import lpips" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :clone_blip
echo Installing requirements for CodeFormer...
%PYTHON% -m pip install -r repositories\CodeFormer\requirements.txt --prefer-binary >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :clone_blip
goto :show_stdout_stderr

:clone_blip
if exist repositories\BLIP goto :check_model
echo Cloning BLIP repository...
%GIT% clone https://github.com/salesforce/BLIP.git repositories\BLIP >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 goto :show_stdout_stderr
%GIT% -C repositories/BLIP checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9 >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% NEQ 0 goto :show_stdout_stderr

:check_model
dir "models\ldm\stable-diffusion-v1\model.ckpt" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_gfpgan
echo Stable Diffusion model not found.
goto :show_stdout_stderr

:check_gfpgan
dir "GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo GFPGAN not found.
echo Face fixing feature will not work.

:launch
echo Launching webui.py...
%PYTHON% webui.py %COMMANDLINE_ARGS%
pause
exit /b

:show_stdout_stderr

echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type tmp\stderr.txt

:endofscript

echo.
echo Launch unsuccessful. Exiting.
pause