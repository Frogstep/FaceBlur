@echo off
rem Starts the FaceBlur web API for the phone app (see README, "Web API"). Extra options are passed on,
rem e.g.  FaceBlurServer.bat --port 9000
cd /d "%~dp0"
".venv\Scripts\python.exe" -m faceblur serve %*
pause
