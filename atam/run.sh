#!/usr/bin/env sh
# ATAM evaluation of the three engines (ewma, garch_t, realized_caviar).
#   ./run.sh                              every part, reusing cached engine runs (atam/work)
#   ./run.sh --fresh                      run every engine again
#   ./run.sh --only robustness,scorecard  some parts only (see source/run_atam.py)
# ewma and garch_t need their programs in bin/ (./build.sh in each module); realized_caviar is
# built (or incrementally rebuilt) here.
set -e
cd "$(dirname "$0")"
python3 -c "import matplotlib, numpy" 2>/dev/null || python3 -m pip install -r requirements.txt
for m in ewma/bin/ewma_risk garch_t/bin/intraday_risk; do
  [ -x "../$m" ] || { echo "../$m not found: run ./build.sh in that module first"; exit 1; }
done
# realized_caviar compiles sources of ewma and garch_t: rebuild it (incremental) so that a change in
# any of the three modules always reaches the evaluation
if [ -f ../realized_caviar/build/CMakeCache.txt ]; then
  cmake --build ../realized_caviar/build --parallel
else
  ../realized_caviar/build.sh
fi
python3 source/run_atam.py "$@"
echo
echo "Done. Start with results/report.txt and the figures in results/figures/."
