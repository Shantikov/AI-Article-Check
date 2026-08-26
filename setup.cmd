@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
set "setup_exit=%ERRORLEVEL%"
echo.
if not "%setup_exit%"=="0" echo Setup failed with exit code %setup_exit%.
pause
exit /b %setup_exit%
