@echo off
rem Runs the EWMA risk engine on the four scenarios:
rem   1. synthetic market, clean feed            -> results\synthetic
rem   2. synthetic market, faulty feed, DQ on    -> results\synthetic_degraded
rem   3. synthetic market, faulty feed, DQ off   -> results\synthetic_degraded_nodq
rem   4. real B3 data                            -> results\b3_real
setlocal
cd /d "%~dp0"
if not exist "bin\ewma_risk.exe" (
  echo bin\ewma_risk.exe not found. Run build.bat first.
  goto :end
)
echo ====================================================================
echo  1/4  Synthetic market: 15 stocks, 150 sessions, options, clean feed
echo ====================================================================
"bin\ewma_risk.exe" --data synthetic
if errorlevel 1 goto :failed
echo.
echo ====================================================================
echo  2/4  Same market, faulty feed, data-quality layer ON
echo ====================================================================
"bin\ewma_risk.exe" --data synthetic --prices ..\data\synthetic\prices_15m_degraded.csv
if errorlevel 1 goto :failed
echo.
echo ====================================================================
echo  3/4  Same market, faulty feed, data-quality layer OFF
echo ====================================================================
"bin\ewma_risk.exe" --data synthetic --prices ..\data\synthetic\prices_15m_degraded.csv --no-dq
if errorlevel 1 goto :failed
echo.
echo ====================================================================
echo  4/4  Real B3 data: 15 stocks, 87 sessions
echo ====================================================================
"bin\ewma_risk.exe" --data b3_real
if errorlevel 1 goto :failed
echo.
echo Done. Results in results\ (start with the report.txt in each folder).
goto :end
:failed
echo.
echo *** Failed: see the message above. Is the "data" folder next to ewma? ***
:end
if not defined NOPAUSE pause
