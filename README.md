# intraday-risk-engine

Low-latency streaming engine for intraday risk.

## Modules

| Folder | What it is |
|---|---|
| [`common/`](common/) | code shared by the three engines: the common model interface, the RiskMetrics baseline and the jump filter |
| [`ewma/`](ewma/) | portfolio engine with the EWMA models (EWMA-N (ref), EWMA-N, EWMA-t) and a data-quality layer |
| [`garch_t/`](garch_t/) | portfolio engine: GARCH(1,1)-t per asset, dynamic correlation, EVT tail, option book by Monte Carlo |
| [`realized_caviar/`](realized_caviar/) | single-instrument engine: Realized-CAViaR against matched EWMA-N, EWMA-t and GARCH-t baselines |
| [`data/`](data/) | the shared datasets (synthetic market, B3 sample) and the tools that degrade and serve them |
| [`atam/`](atam/) | the architecture evaluation (ATAM) of the three engines: scenarios, measurements, scorecard, findings |
| [`architecture/`](architecture/) | diagrams of the system |

Each engine builds with its own `build.bat` / `build.sh` (CMake, a C++17 compiler) and runs its tests
with `ctest`; each README lists its options and its results.

## Dashboard

[`dashboard/`](dashboard/) is a static site (no build step) that explains the models, shows the comparison and runs the C++ engine in the browser via WebAssembly. It is available in Portuguese and English. Run it locally with `python -m http.server 8080 -d dashboard`; it is published to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`. See [dashboard/README.md](dashboard/README.md).
