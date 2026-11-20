@echo off
rem Builds the zip to copy to another PC (dist\FaceBlur-<date>.zip). Options are passed on:
rem   make_dist.bat          build the zip
rem   make_dist.bat --lock   regenerate install\requirements-*.txt first (needs uv)
setlocal
cd /d "%~dp0"
rem let --lock find the uv that install.bat downloaded if uv is not on PATH
if exist "tools\uv\uv.exe" set "PATH=%~dp0tools\uv;%PATH%"
".venv\Scripts\python.exe" install\make_dist.py %*
set "RESULT=%errorlevel%"
if not "%RESULT%"=="0" echo. & echo make_dist failed with code %RESULT%.
pause
exit /b %RESULT%
