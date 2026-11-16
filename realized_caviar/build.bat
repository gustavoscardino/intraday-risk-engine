@echo off
setlocal
cmake -S "%~dp0source" -B "%~dp0build" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%~dp0build" --config Release --parallel
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0build" -C Release --output-on-failure
exit /b %errorlevel%
