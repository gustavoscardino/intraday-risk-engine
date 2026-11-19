#!/usr/bin/env bash
# Compiles the matched-comparison engine (realized_caviar + the ewma / garch_t sources it uses)
# to WebAssembly for the dashboard: dashboard/engine/rcaviar.{js,wasm,data}.
# Needs Emscripten (emsdk) activated in this shell:  source <emsdk>/emsdk_env.sh
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
command -v em++ >/dev/null || { echo "em++ not found: activate emsdk first (source <emsdk>/emsdk_env.sh)"; exit 1; }
mkdir -p dashboard/engine
em++ -O3 -std=c++20 -fwasm-exceptions -Wall -Wextra \
  -I realized_caviar/source/include -I garch_t/source/include -I ewma/source/include \
  realized_caviar/source/wasm/engine_api.cpp \
  realized_caviar/source/src/model.cpp realized_caviar/source/src/data.cpp \
  realized_caviar/source/src/shared_data.cpp realized_caviar/source/src/replay.cpp \
  garch_t/source/src/math.cpp garch_t/source/src/optimize.cpp garch_t/source/src/garch.cpp \
  garch_t/source/src/risk.cpp garch_t/source/src/backtest.cpp garch_t/source/src/market_data.cpp \
  ewma/source/src/math.cpp ewma/source/src/ewma.cpp ewma/source/src/seasonality.cpp \
  -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEngine -sENVIRONMENT=web,worker \
  -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=1 \
  --preload-file data/synthetic/assets.csv@/data/synthetic/assets.csv \
  --preload-file data/synthetic/prices_15m.csv@/data/synthetic/prices_15m.csv \
  --preload-file data/b3_real/assets.csv@/data/b3_real/assets.csv \
  --preload-file data/b3_real/prices_15m.csv@/data/b3_real/prices_15m.csv \
  -o dashboard/engine/rcaviar.js
echo "built dashboard/engine/rcaviar.{js,wasm,data}"
