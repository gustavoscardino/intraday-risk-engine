#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$project_dir/build.sh"
"$project_dir/build/realized_caviar" --data "$project_dir/../data/synthetic" --ticker XPET3 --out "$project_dir/results/shared_synthetic_XPET3" "$@"
"$project_dir/build/realized_caviar" --data "$project_dir/../data/b3_real" --ticker PETR4 --out "$project_dir/results/shared_b3_real_PETR4" "$@"
