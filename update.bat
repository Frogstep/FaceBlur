@echo off
rem Updates FaceBlur to the newest release from GitHub, keeping models, projects and settings.
rem   update.bat                  install the newest version (if newer)
rem   update.bat --check          only check for a newer version
rem   update.bat FaceBlur-X.zip   install from a zip file (or drag the zip onto this file)
rem   update.bat --rollback       go back to the previous version
rem The update replaces this file too: everything is in one block, which cmd reads before running it.
cd /d "%~dp0"
(
    ".venv\Scripts\python.exe" install\update.py %*
    pause
    exit /b
)
