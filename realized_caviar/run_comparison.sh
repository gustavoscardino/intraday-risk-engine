#!/usr/bin/env bash
# Builds, tests and runs the matched comparison on every stock of data/synthetic and data/b3_real
# (Realized-CAViaR, EWMA-N, EWMA-t, GARCH-t; fixed estimation window; Diebold-Mariano tests).
# Consolidated tables: realized_caviar/comparison/ (start with report.txt).
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$project_dir/build.sh"
python3 "$project_dir/tools/compare_all.py" "$@"
