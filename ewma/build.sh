#!/usr/bin/env sh
# Rebuilds the programs (copied to bin/) and runs the automated tests.
# Needs CMake >= 3.16 and a C++17 compiler (g++ or clang++).
set -e
cd "$(dirname "$0")"
GEN=""
if command -v ninja >/dev/null 2>&1; then GEN="-G Ninja"; fi
cmake -S source -B source/build $GEN -DCMAKE_BUILD_TYPE=Release
cmake --build source/build --parallel
(cd source/build && ctest --output-on-failure)
echo "Done: programs updated in bin/ and all tests passed."
