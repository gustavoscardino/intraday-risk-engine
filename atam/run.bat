@echo off
rem ATAM evaluation of the three engines (ewma, garch_t, realized_caviar).
rem   run.bat                              every part, reusing cached engine runs (atam\work)
rem   run.bat --fresh                      run every engine again
rem   run.bat --only robustness,scorecard  some parts only (see source\run_atam.py)
rem ewma and garch_t use their versioned programs in bin\; realized_caviar is built (or rebuilt) here.
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer was not found in the PATH.
  goto :failed
)
python -c "import matplotlib, numpy" >nul 2>nul
if errorlevel 1 (
  echo Installing matplotlib and numpy...
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)
rem realized_caviar compiles sources of ewma and garch_t: rebuild it (incremental, fast when nothing
rem changed) so that a change in any of the three modules always reaches the evaluation
if exist "..\realized_caviar\build\CMakeCache.txt" (
  cmake --build "..\realized_caviar\build" --config Release --parallel
) else (
  echo Building realized_caviar...
  call "..\realized_caviar\build.bat"
)
if errorlevel 1 goto :failed
python -X utf8 source\run_atam.py %*
if errorlevel 1 goto :failed
echo.
echo Done. Start with results\report.txt and the figures in results\figures\.
goto :end
:failed
echo.
echo *** Failed: see the message above. ***
if not defined NOPAUSE pause
exit /b 1
:end
if not defined NOPAUSE pause
