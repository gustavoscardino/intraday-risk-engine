@echo off
setlocal
call "%~dp0build.bat"
if errorlevel 1 exit /b 1
set "rcaviar_exe=%~dp0build\realized_caviar.exe"
if exist "%~dp0build\Release\realized_caviar.exe" set "rcaviar_exe=%~dp0build\Release\realized_caviar.exe"
"%rcaviar_exe%" --out "%~dp0results\synthetic" %*
exit /b %errorlevel%
