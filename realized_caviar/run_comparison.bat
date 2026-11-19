@echo off
rem Builds, tests and runs the matched comparison on every stock of data\synthetic and data\b3_real
rem (Realized-CAViaR, EWMA-N, EWMA-t, GARCH-t; fixed estimation window; Diebold-Mariano tests).
rem Consolidated tables: realized_caviar\comparison\ (start with report.txt).
setlocal
call "%~dp0build.bat"
if errorlevel 1 exit /b 1
python "%~dp0tools\compare_all.py" %*
exit /b %errorlevel%
