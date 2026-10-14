@echo off
rem Rebuilds the programs (copied to bin\) and runs the automated tests.
rem Needs CMake and a C++17 compiler. The easiest option on Windows is WinLibs (GCC + CMake + Ninja):
rem   winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --source winget
rem When WinLibs is installed it is put first in the PATH, so an older MinGW elsewhere in the
rem PATH (C:\MinGW, GCC 6) is not picked up by mistake.
setlocal
cd /d "%~dp0"
set "WINLIBS=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
if exist "%WINLIBS%\g++.exe" set "PATH=%WINLIBS%;%PATH%"
where cmake >nul 2>nul
if errorlevel 1 (
  echo CMake not found. Install WinLibs with:
  echo   winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --source winget
  goto :end
)
set "GEN="
where ninja >nul 2>nul
if not errorlevel 1 set "GEN=-G Ninja"
cmake -S source -B source\build %GEN% -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 goto :failed
cmake --build source\build --config Release
if errorlevel 1 goto :failed
pushd source\build
ctest -C Release --output-on-failure
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" goto :failed
echo.
echo Done: programs updated in bin\ and all tests passed.
goto :end
:failed
echo.
echo *** Failed: see the messages above. If you switched compilers, delete source\build and run again. ***
:end
if not defined NOPAUSE pause
