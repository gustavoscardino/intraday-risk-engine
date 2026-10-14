# Realized-CAViaR learning implementation

A small C++20 experiment following `../DECISIONS.md`, arranged like `garch_t`
(`source/include`, `source/src`, `source/apps`, and `source/tests`). It reuses the
existing engine's math and coverage-test code; the GARCH engine is unchanged.
No external C++ libraries are required.

## Run

From the repository root, with CMake and a C++20 compiler installed:

```sh
cmake -S realized_caviar/source -B realized_caviar/build -DCMAKE_BUILD_TYPE=Release
cmake --build realized_caviar/build --parallel
ctest --test-dir realized_caviar/build --output-on-failure
./realized_caviar/build/realized_caviar --out realized_caviar/results/synthetic
```

Or run `./realized_caviar/run.sh`. The default is a seeded, single-instrument
synthetic price history with volatility clustering and Student-t shocks. There
is no external market-data download. `--help` lists the CLI options. See the shared-data mode below to use the same
CSV files as `garch_t`.

Outputs:

- `forecasts.csv`: every out-of-sample minute forecast, its future target,
  realized variance, quantile, long-position VaR for 100,000 monetary units,
  exception, and whether it was selected for non-overlapping evaluation.
- `report.txt`: fitted coefficients and optimizer diagnostics, training loss
  versus a constant-quantile candidate, exception rate, Kupiec,
  Christoffersen independence/conditional coverage, mean pinball loss, fitting
  time, update mean/p50/p95/p99 latency, throughput, and filter state size.

Generated results are ignored by Git. Times are integer seconds in a continuous
sample timeline. The synthetic distribution is seeded, but exact generated
values may differ between C++ standard-library implementations.

## Model and timing

Prices give decimal log returns `r_j = log(P_j) - log(P_(j-1))` every 5 seconds.
At each minute close, take the last 60 returns:

```
R_t  = sum(r_j)
RV_t = sum(r_j * r_j)
q_(t+1) = beta0 + beta1*q_t + beta2*abs(R_t) + beta3*sqrt(RV_t)
```

The resulting quantile targets the next 5-minute log return, at tail probability
`tau = 0.01` (99% VaR). The recurrence advances once per minute; `t+1` denotes
the next forecast update, while the target horizon remains 5 minutes. This
cadence convention is part of this experiment.

Offline estimation minimizes the exact mean pinball loss:
`(y-q) * (tau - indicator(y < q))`. It uses a small deterministic multi-start
Nelder-Mead optimizer because the objective has kinks. Returns are scaled during
fitting for numerical conditioning. The empirical training-tail quantile seeds
the recursion; a constant quantile is always retained as a candidate. Inspect
`FitResult::converged`: a valid best candidate can be returned after reaching
the iteration limit, and convergence does not establish a global optimum.

The learning implementation imposes `beta0,beta2,beta3 <= 0` and
`0 <= beta1 <= 0.995`. Starting from a negative quantile, these restrictions keep
forecasts negative and the recurrence stable. They are additional modeling
choices beyond the equation in DECISIONS.md, and may limit the fitted model.

In the high-frequency mode, the first half of the price timeline is training. Only training windows whose
**future target ends before or at the cutoff** enter fitting. After fitting,
replay rebuilds the quantile state through all observed features to the cutoff,
including the intervening windows whose labels were excluded. NaN training
targets advance the recursion through their observed features but contribute
no loss; infinite targets are rejected. Future target
returns are used only for scoring, never in the filter update.

High-frequency operational forecasts overlap. Primary backtesting selects every fifth forecast
so target intervals are adjacent and non-overlapping. All minute forecasts are
still written to CSV. Coverage p-values use the existing engine's asymptotic
tests; short samples and sparse 1% exceptions can make these uninformative.

For a long exposure `E`, monetary loss VaR is `-E * expm1(q)`, converting the log
quantile into a simple-return loss. Shorts require an upper-tail model and are
rejected by this helper.

## Own data

```sh
./realized_caviar/build/realized_caviar --prices prices.csv --out realized_caviar/results/custom
```

Use exactly `timestamp,price` with integer seconds and positive finite prices.
Prices must be in strict timestamp order, with exactly 5 seconds between rows.
Duplicates, out-of-order samples, invalid values, and missing/stale samples stop
the run with an error. Split sessions into separate continuous files; overnight
gaps are unsupported. Valid extreme returns are retained. At least 100 fully
labeled training windows are required, and meaningful 1% tail evaluation needs
substantially more observations.

## Shared data with garch_t

This mode reads the existing `data/synthetic/` and `data/b3_real/` folders
without converting or modifying them. Each run selects one instrument from
`assets.csv` and reads its prices from `prices_15m.csv` using the existing
`garch_t` loader and return calculation. The full asset grid is loaded so missing
observations are detected consistently with the original engine.

Run both example instruments with `./realized_caviar/run_shared.sh`, or:

```sh
./realized_caviar/build/realized_caviar --data data/synthetic --ticker XPET3 --out realized_caviar/results/shared_synthetic_XPET3
./realized_caviar/build/realized_caviar --data data/b3_real --ticker PETR4 --out realized_caviar/results/shared_b3_real_PETR4
```

The existing data contain **15-minute bars**, so this is a separate temporal
configuration, explicitly agreed for comparison experiments:

| Setting | Shared-data mode |
|---|---|
| Sampling | 15 minutes |
| Forecast horizon / update | 15 minutes / 15 minutes |
| Information / realized window | Previous hour: four 15-minute returns |
| Aggregate input | Sum of those four returns |
| Realized variance input | Sum of their squares |
| Target | Next single 15-minute return |
| Confidence | 99% |

No five-second prices are interpolated. This is a coarse realized measure and
does not implement the original five-second/five-minute configuration.

Return conventions match `garch_t`: the first bar of a session uses
`log(close/open)`; subsequent bars use intraday close-to-close returns. Overnight
gaps are excluded. All four feature returns must lie within one session on consecutive grid
slots. Feature windows touching missing prices, invalid returns, or gaps are
skipped, including returns that span a missing price. Forecasts never target
the next session. If an expected intraday target is missing, the forecast is
still emitted and its observed features advance the quantile state; the target
is NaN and the forecast is excluded from scoring. Missing future data therefore
cannot decide whether an observed feature update occurs.
The quantile state is carried across sessions and skipped windows; no update is
performed when no complete feature/target window is available. This is a replay
convention, not an overnight risk model.

Training uses the same default initial number of sessions as `garch_t`: half
of available sessions, capped at 60. Thus synthetic uses 60 of 150 sessions;
B3 uses 30 of 60. `--window-days N` overrides this initial training period.
Parameters are still fitted **once**; this does not replicate `garch_t`'s rolling
daily refits.

CSV feature columns are named `aggregate_return` and `realized_variance` in both
modes. Shared-data timestamps are seconds since 1970 on the exchange-local
wall-clock timeline, derived from `garch_t::TimePoint::key()`; they are not UTC
instants. Every shared-data target is non-overlapping. Independence tests pool
transition counts only between adjacent target intervals, excluding artificial
transitions across sessions and missing windows.

This prepares a later fair comparison: run all models for the same selected
instrument, score the intersection of target timestamps, and align their
calibration schedules and exposures. The old GARCH portfolio reports cover
multiple assets and more target windows, so they cannot directly rank the new
single-instrument results. The one-hour feature warmup excludes early-session
forecasts and the next-day target at each close.

Stored example outputs from the shared-data run:

| Instrument | Training sessions | OOS targets | Exceptions | Rate | Kupiec p |
|---|---:|---:|---:|---:|---:|
| Synthetic XPET3 | 60 | 2,156 | 13 | 0.60% | 0.0455 |
| B3 PETR4 | 30 | 720 | 19 | 2.64% | 0.000242 |

The synthetic run is somewhat conservative; the real-data run underestimates
risk. Both coverage tests reject the intended 1% exception rate at the 5% level.
These are learning results for one instrument each, without a matched GARCH
comparison yet.

## Library

```cpp
#include "rcaviar/model.hpp"
// training contains observed backward-window features and their future targets.
auto fitted = rcaviar::fit(training);
rcaviar::Filter filter(fitted.params, fitted.last_quantile);
// Immediately following training's final forecast update:
double q = filter.update(observed_return_5m, observed_rv_5m);
double var = rcaviar::long_var(long_exposure, q);
```

The filter holds only four coefficients and one quantile. `update` does no
allocation or optimization. Invalid features throw before changing the state.
History collection and fitting are separate from this online filter.

## Limits and reference

This is a learning experiment, not an empirical claim of superiority. It does
not yet compare all three models on matched forecast cadences, process multiple
instruments in one run, estimate uncertainty, perform rolling refits, or
implement ES. Shared-data mode excludes session boundaries; high-frequency
mode still requires a continuous timeline. Latency includes clock overhead and measures model
updates only, excluding CSV, window construction, and offline fitting.

The recursive quantile and quantile-loss approach follows
[Engle and Manganelli's CAViaR paper](https://pages.stern.nyu.edu/~rengle/CAVIAR.pdf).
The realized-variance term here is the explicit specification in this repo's
DECISIONS.md; this experiment does not reproduce a particular published
Realized-CAViaR estimator.
