# intraday-risk-engine

Low-latency streaming engine for intraday risk.

## Dashboard

[`dashboard/`](dashboard/) is a static site (no build step) that explains the models, shows the comparison and runs the C++ engine in the browser via WebAssembly. It is available in Portuguese and English. Run it locally with `python -m http.server 8080 -d dashboard`; it is published to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`. See [dashboard/README.md](dashboard/README.md).
