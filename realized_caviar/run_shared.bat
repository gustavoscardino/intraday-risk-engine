@echo off
setlocal
call "%~dp0build.bat"
if errorlevel 1 exit /b 1
set "rcaviar_exe=%~dp0build\realized_caviar.exe"
if exist "%~dp0build\Release\realized_caviar.exe" set "rcaviar_exe=%~dp0build\Release\realized_caviar.exe"
"%rcaviar_exe%" --data "%~dp0..\data\synthetic" --ticker XPET3 --out "%~dp0results\shared_synthetic_XPET3" %*
if errorlevel 1 exit /b 1
"%rcaviar_exe%" --data "%~dp0..\data\b3_real" --ticker PETR4 --out "%~dp0results\shared_b3_real_PETR4" %*
exit /b %errorlevel%
