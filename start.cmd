@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set "start_exit=%ERRORLEVEL%"
echo.
if not "%start_exit%"=="0" echo Server stopped with exit code %start_exit%.
pause
exit /b %start_exit%
