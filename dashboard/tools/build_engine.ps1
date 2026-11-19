# Compiles the matched-comparison engine (realized_caviar + the ewma / garch_t sources it uses)
# to WebAssembly for the dashboard: dashboard\engine\rcaviar.{js,wasm,data}.
# Needs Emscripten (emsdk). Usage, from any folder:
#   powershell -ExecutionPolicy Bypass -File dashboard\tools\build_engine.ps1 [-Emsdk C:\path\to\emsdk]
param([string]$Emsdk = "$env:USERPROFILE\emsdk")
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not (Get-Command em++ -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path "$Emsdk\emsdk_env.ps1")) { throw "em++ not found and no emsdk at $Emsdk (pass -Emsdk)" }
    . "$Emsdk\emsdk_env.ps1" | Out-Null
}
Set-Location $repo
New-Item -ItemType Directory -Force dashboard\engine | Out-Null
$sources = @(
    'realized_caviar/source/wasm/engine_api.cpp',
    'realized_caviar/source/src/model.cpp', 'realized_caviar/source/src/data.cpp',
    'realized_caviar/source/src/shared_data.cpp', 'realized_caviar/source/src/replay.cpp',
    'garch_t/source/src/math.cpp', 'garch_t/source/src/optimize.cpp', 'garch_t/source/src/garch.cpp',
    'garch_t/source/src/risk.cpp', 'garch_t/source/src/backtest.cpp', 'garch_t/source/src/market_data.cpp',
    'ewma/source/src/math.cpp', 'ewma/source/src/ewma.cpp', 'ewma/source/src/seasonality.cpp')
$data = @('synthetic/assets.csv', 'synthetic/prices_15m.csv', 'b3_real/assets.csv', 'b3_real/prices_15m.csv') |
    ForEach-Object { '--preload-file'; "data/$_@/data/$_" }
& em++ -O3 -std=c++20 -fwasm-exceptions -Wall -Wextra `
    -I realized_caviar/source/include -I garch_t/source/include -I ewma/source/include `
    @sources -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEngine '-sENVIRONMENT=web,worker' `
    -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=1 @data -o dashboard/engine/rcaviar.js
if ($LASTEXITCODE) { throw "em++ failed" }
Write-Host "built dashboard\engine\rcaviar.{js,wasm,data}"
