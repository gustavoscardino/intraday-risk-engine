#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$project_dir/source" -B "$project_dir/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$project_dir/build" --parallel
ctest --test-dir "$project_dir/build" --output-on-failure
"$project_dir/build/realized_caviar" --data "$project_dir/../data/synthetic" --ticker XPET3 --out "$project_dir/results/shared_synthetic_XPET3"
"$project_dir/build/realized_caviar" --data "$project_dir/../data/b3_real" --ticker PETR4 --out "$project_dir/results/shared_b3_real_PETR4"
