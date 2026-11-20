@echo off
rem Publishes a new FaceBlur version as a GitHub release (see install\publish.py).
rem   publish.bat              build, ask for a note, upload
rem   publish.bat --dry-run    build the zip and version.json only
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" install\publish.py %*
set "RESULT=%errorlevel%"
if not "%RESULT%"=="0" echo. & echo publish failed with code %RESULT%.
pause
exit /b %RESULT%
