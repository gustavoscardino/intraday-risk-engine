#!/usr/bin/env sh
# Runs the risk engine on both datasets (synthetic and real B3 data).
# Results: results/synthetic and results/b3_real (report.txt + CSV files).
set -e
cd "$(dirname "$0")"
if [ ! -x bin/intraday_risk ]; then
  echo "bin/intraday_risk not found: run ./build.sh first"
  exit 1
fi
echo "=== 1/2  Synthetic market: 15 fictitious stocks, 150 sessions, options ==="
./bin/intraday_risk --data synthetic
echo
echo "=== 2/2  Real B3 data: 15 stocks, 87 sessions ==="
./bin/intraday_risk --data b3_real
echo
echo "Done. Results in results/synthetic and results/b3_real (start with report.txt)."
