@echo off
rem Runs the risk engine on both datasets (synthetic and real B3 data).
rem Results: results\synthetic and results\b3_real (report.txt + CSV files).
setlocal
cd /d "%~dp0"
if not exist "bin\intraday_risk.exe" (
  echo bin\intraday_risk.exe not found. Run build.bat first.
  goto :end
)
echo ====================================================================
echo  1/2  Synthetic market: 15 fictitious stocks, 150 sessions, options
echo ====================================================================
"bin\intraday_risk.exe" --data synthetic
if errorlevel 1 goto :failed
echo.
echo ====================================================================
echo  2/2  Real B3 data: 15 stocks, 60 sessions
echo ====================================================================
"bin\intraday_risk.exe" --data b3_real
if errorlevel 1 goto :failed
echo.
echo Done. Results in results\synthetic and results\b3_real
echo (start with the report.txt in each folder).
goto :end
:failed
echo.
echo *** Failed: see the message above. Is the "data" folder next to garch_t? ***
:end
pause
