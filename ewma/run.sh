#!/usr/bin/env sh
# Runs the EWMA risk engine on the four scenarios (clean feed, faulty feed with and without the
# data-quality layer, real B3 data). Results: results/<scenario>/report.txt + CSV files.
set -e
cd "$(dirname "$0")"
if [ ! -x bin/ewma_risk ]; then
  echo "bin/ewma_risk not found: run ./build.sh first"
  exit 1
fi
echo "=== 1/4  Synthetic market, clean feed ==="
./bin/ewma_risk --data synthetic
echo
echo "=== 2/4  Same market, faulty feed, data-quality layer ON ==="
./bin/ewma_risk --data synthetic --prices ../data/synthetic/prices_15m_degraded.csv
echo
echo "=== 3/4  Same market, faulty feed, data-quality layer OFF ==="
./bin/ewma_risk --data synthetic --prices ../data/synthetic/prices_15m_degraded.csv --no-dq
echo
echo "=== 4/4  Real B3 data ==="
./bin/ewma_risk --data b3_real
echo
echo "Done. Results in results/ (start with report.txt in each folder)."
