@echo off
rem Builds and tests. WinLibs (winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e) is put
rem first in the PATH when installed, so an older MinGW elsewhere in the PATH is not used.
setlocal
set "WINLIBS=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
if exist "%WINLIBS%\g++.exe" set "PATH=%WINLIBS%;%PATH%"
set "GEN="
where ninja >nul 2>nul
if not errorlevel 1 set "GEN=-G Ninja"
cmake -S "%~dp0source" -B "%~dp0build" %GEN% -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%~dp0build" --config Release --parallel
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0build" -C Release --output-on-failure
exit /b %errorlevel%
