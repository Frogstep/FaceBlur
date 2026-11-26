@echo off
rem FaceBlur installer. Downloads Python, the Python packages, the model weights and
rem FFmpeg into this folder (nothing is installed system-wide). Safe to run again: it
rem skips what is already there and resumes interrupted model downloads.
rem
rem   install.bat        GPU build if an NVIDIA driver is found, else CPU build
rem   install.bat gpu    force the NVIDIA (CUDA 13) build
rem   install.bat cpu    force the CPU-only build (much smaller download)
rem
rem To uninstall, delete this folder.

setlocal
cd /d "%~dp0"

set "UV_VERSION=0.11.15"
set "UV_SHA256=04b98d414a9000e25e5e0e7c9f53749e66b790cdaffc582829e6f58c544ee11c"
set "SAM2=SAM-2 @ https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.zip"

rem keep everything inside this folder
set "UV=%~dp0tools\uv\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%~dp0tools\python"
set "UV_CACHE_DIR=%~dp0tools\uv-cache"
set "UV_LINK_MODE=copy"
set "UV_HTTP_TIMEOUT=300"

rem ------------------------------------------------------------ GPU or CPU
set "VARIANT=%~1"
if "%VARIANT%"=="" (
    set "VARIANT=cpu"
    nvidia-smi >nul 2>&1 && set "VARIANT=gpu"
)
if /i not "%VARIANT%"=="gpu" if /i not "%VARIANT%"=="cpu" (
    echo Usage: install.bat [gpu^|cpu]
    exit /b 2
)
rem exact local versions (+cu130 / +cpu) so switching builds really replaces PyTorch
if /i "%VARIANT%"=="gpu" (
    set "TORCH=torch==2.11.0+cu130 torchvision==0.26.0+cu130"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
    set "OTHER_ORT=onnxruntime"
    echo Installing the NVIDIA GPU build: about 4 GB to download.
) else (
    set "TORCH=torch==2.11.0+cpu torchvision==0.26.0+cpu"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "OTHER_ORT=onnxruntime-gpu"
    echo Installing the CPU-only build: about 1 GB to download. Processing will be slow.
)
echo.

rem ------------------------------------------------------------ uv
if exist "%UV%" (
    echo [1/5] uv already present
) else (
    call :get_uv
)
if not exist "%UV%" goto :fail
rem uv, Python and FFmpeg are not code-signed, so Smart App Control or an App Control
rem (Device Guard) policy refuses to run them: say so instead of failing halfway
"%UV%" --version >nul 2>&1
if errorlevel 1 goto :blocked

rem ------------------------------------------------------------ Python + venv
echo [2/5] Python 3.12 and virtual environment
if not exist ".venv\Scripts\python.exe" (
    "%UV%" venv --python 3.12 --python-preference only-managed .venv
    if errorlevel 1 goto :fail
)

rem ------------------------------------------------------------ packages
echo [3/5] Python packages (PyTorch is large; this takes a while)
rem onnxruntime and onnxruntime-gpu share one module: remove the other build if present
"%UV%" pip uninstall --python .venv %OTHER_ORT% >nul 2>&1
"%UV%" pip install --python .venv %TORCH% --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail
"%UV%" pip install --python .venv -r install\requirements-windows-%VARIANT%.txt
if errorlevel 1 goto :fail
rem SAM 2 from source: no CUDA extension (optional, needs a compiler), and no build
rem isolation so it does not download a second PyTorch just to build
set "SAM2_BUILD_CUDA=0"
"%UV%" pip install --python .venv --no-deps --no-build-isolation "%SAM2%"
if errorlevel 1 goto :fail

rem PyTorch and onnxruntime need Microsoft's Visual C++ Runtime in Windows itself,
rem which a fresh Windows may lack (or have in a too old version)
".venv\Scripts\python.exe" -c "import torch, onnxruntime" >nul 2>&1
if errorlevel 1 call :get_vcredist
if errorlevel 1 goto :fail

rem ------------------------------------------------------------ models + FFmpeg
echo [4/5] Model weights and FFmpeg
if /i "%VARIANT%"=="gpu" (
    ".venv\Scripts\python.exe" -m faceblur.setup_models --gpu
) else (
    ".venv\Scripts\python.exe" -m faceblur.setup_models --cpu
)
if errorlevel 1 goto :fail

rem the package cache is only needed during install
rmdir /s /q "%UV_CACHE_DIR%" >nul 2>&1

rem ------------------------------------------------------------ check
echo [5/5] Checking the installation
echo.
".venv\Scripts\python.exe" -m faceblur.setup_models --check
echo.
echo Done. Start FaceBlur with FaceBlur.bat (or drag a video onto it).
pause
exit /b 0

rem ------------------------------------------------------------ subroutines
:get_uv
rem curl.exe, tar.exe and certutil ship with Windows 10 (1803+) and 11
echo [1/5] Downloading uv %UV_VERSION% (Python installer)
set "UV_ZIP=%~dp0tools\uv.zip"
if not exist "%~dp0tools\uv" mkdir "%~dp0tools\uv"
"%SystemRoot%\System32\curl.exe" -fL --retry 3 -o "%UV_ZIP%" "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 exit /b 1
"%SystemRoot%\System32\certutil.exe" -hashfile "%UV_ZIP%" SHA256 | "%SystemRoot%\System32\findstr.exe" /i /x "%UV_SHA256%" >nul
if errorlevel 1 (
    echo uv download is corrupted
    del "%UV_ZIP%"
    exit /b 1
)
"%SystemRoot%\System32\tar.exe" -xf "%UV_ZIP%" -C "%~dp0tools\uv"
del "%UV_ZIP%"
exit /b 0

:get_vcredist
echo.
echo PyTorch needs the Microsoft Visual C++ Runtime, which is missing or too old.
echo Installing Microsoft's official package: Windows will ask for permission.
set "VCREDIST=%~dp0tools\vc_redist.x64.exe"
"%SystemRoot%\System32\curl.exe" -fL --retry 3 -o "%VCREDIST%" "https://aka.ms/vc14/vc_redist.x64.exe"
if errorlevel 1 exit /b 1
rem run it only if it is really signed by Microsoft; RunAs shows the UAC prompt.
rem An empty PSModulePath makes Windows PowerShell use its own modules even when
rem this script was started from PowerShell 7.
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = Get-AuthenticodeSignature -LiteralPath $env:VCREDIST;" ^
  "if ($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') { Write-Host 'The download is not signed by Microsoft.'; exit 1 };" ^
  "try { $p = Start-Process -FilePath $env:VCREDIST -ArgumentList '/install','/passive','/norestart' -Verb RunAs -Wait -PassThru } catch { Write-Host 'Permission was not given.'; exit 1 };" ^
  "if (@(0, 1638, 3010) -notcontains $p.ExitCode) { Write-Host ('The runtime installer failed with code ' + $p.ExitCode); exit 1 }"
set "VCRESULT=%errorlevel%"
del "%VCREDIST%" >nul 2>&1
if not "%VCRESULT%"=="0" exit /b 1
".venv\Scripts\python.exe" -c "import torch, onnxruntime"
if errorlevel 1 (
    echo PyTorch still does not load. If the runtime installer asked for a restart,
    echo restart Windows and run install.bat again.
    exit /b 1
)
echo Visual C++ Runtime installed.
exit /b 0

:blocked
echo.
"%SystemRoot%\System32\reg.exe" query "HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy" /v VerifiedAndReputablePolicyState 2>nul | "%SystemRoot%\System32\findstr.exe" /e "0x1" >nul
if errorlevel 1 (
    echo Windows refused to run "%UV%".
    echo An App Control ^(Device Guard^) policy on this PC blocks programs that are not
    echo code-signed. FaceBlur's tools ^(uv, Python, FFmpeg^) are open source and not signed,
    echo so FaceBlur can't be installed here unless the PC's administrator allows them.
) else (
    echo Smart App Control is on, and it blocks programs that are not code-signed.
    echo FaceBlur's tools ^(uv, Python, FFmpeg^) are open source and not signed, so they
    echo can't run while it is on. To install FaceBlur, turn it off in Windows Security:
    echo App ^& browser control, Smart App Control settings, Off. Then run install.bat again.
)
pause
exit /b 1

:fail
echo.
echo Installation failed (see the messages above). Fix the problem, e.g. the internet
echo connection, and run install.bat again: finished steps are skipped.
pause
exit /b 1
