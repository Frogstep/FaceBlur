@echo off
rem Starts the FaceBlur app. Drag a video or .faceblur.json onto this file to open it directly.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m faceblur gui %*
