# Realized-CAViaR and matched intraday VaR comparison

C++20 library and replay for `../DECISIONS.md`. Runs Realized-CAViaR, EWMA-N
and GARCH-t on **the same instrument, forecast timestamps, future return targets,
confidence, signed exposure and calibration calendar**. The EWMA and GARCH
implementations are compiled from the existing modules; their portfolio engines
and stored reports are unchanged. No external C++ library is needed.

## Build and run

Requires CMake and a C++20 compiler. On Fedora:

```sh
sudo dnf install gcc-c++ cmake ninja-build
# Optional independent estimator validation:
sudo dnf install python3-scipy
# Optional address/undefined-behavior sanitizer runtimes:
sudo dnf install libasan libubsan
```

From the repository root:

```sh
cmake -S realized_caviar/source -B realized_caviar/build -DCMAKE_BUILD_TYPE=Release
cmake --build realized_caviar/build --parallel
ctest --test-dir realized_caviar/build --output-on-failure

# Original configuration: 5s sampling, 5m horizon, forecasts every minute.
./realized_caviar/build/realized_caviar --out realized_caviar/results/synthetic

# Same existing 15m data as the other engines, daily rolling calibration.
./realized_caviar/build/realized_caviar --data data/b3_real --ticker PETR4 \
  --out realized_caviar/results/shared_b3_real_PETR4

# Controlled once-only calibration, short exposure and 95% confidence.
./realized_caviar/build/realized_caviar --data data/synthetic --ticker XPET3 \
  --refit once --exposure -200000 --confidence .95 \
  --out realized_caviar/results/short_XPET3
```

On Windows, `realized_caviar\build.bat` builds and tests with WinLibs GCC (put first in the PATH
when installed: `winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e`). The programs carry a
UTF-8 manifest, so project folders with accents in their names work, as in `ewma` and `garch_t`.

`./realized_caviar/build.sh` builds and tests. `run.sh` additionally runs the
seeded high-frequency example; `run_shared.sh` runs synthetic XPET3 and B3 PETR4.
Equivalent `.bat` scripts are supplied for Windows. Shared-run scripts forward
additional CLI arguments to both runs.

| Option | Meaning | Default |
|---|---|---|
| `--data FOLDER --ticker SYMBOL` | existing shared 15m dataset, one instrument | seeded 5s simulation |
| `--prices FILE` | 5s `timestamp,price` CSV, or alternate 15m CSV with `--data` | built-in simulation / `prices_15m.csv` |
| `--out FOLDER` | output folder | `results/synthetic` |
| `--window-days N` | initial and rolling training sessions in shared mode | half the sessions, capped at 60 |
| `--refit once\|daily` | common calibration schedule for all models | daily shared; once high frequency |
| `--exposure E` | signed monetary exposure, negative for shorts | 100000 |
| `--confidence C` | confidence in (.5, 1) | .99 |
| `--lambda L` | fixes the EWMA decay in (0, 1) | estimated at each calibration |
| `--diurnal on\|off` | shared mode: baselines on returns divided by the U profile | on |
| `--caviar-only` | disable the matched baselines | all three models |
| `--reset-on-gap` | explicit restart of 5s features after missing samples/session gaps | reject gaps |

Invalid values and incompatible options fail explicitly. Real high-frequency
market data must be supplied by the caller; there is no downloader or fabricated
interpolation of 15m bars into 5s prices.

## Timing and comparison

| Setting | High-frequency mode | Shared-data mode |
|---|---|---|
| Input | positive prices every 5s | existing `assets.csv` / `prices_15m.csv` |
| Forecast / update | 5m / 1m | 15m / 15m |
| Realized information window | preceding 60 five-second returns | preceding four 15m returns |
| Training | first half of price timeline | initial N sessions |
| EWMA/GARCH innovation | non-overlapping 5m return within each phase | single 15m return |

High-frequency baselines have **five staggered filters per model**, offset by a
minute. Each filter observes non-overlapping five-minute returns and predicts
its next five-minute return. Selecting the corresponding phase every minute
produces the same targets as CAViaR, without square-root-of-time scaling or an
assumption that a sum of Student-t shocks is Student-t. Each phase is fitted on
its own history. Cross-phase dependence remains; primary backtesting selects
non-overlapping targets. The state-size metric includes all five filters.

Shared mode uses the existing loader and return conventions: first bar
`log(close/open)`, then intraday close-to-close. The full asset grid is loaded,
but only the selected instrument is modeled. No target crosses the session
close. Incomplete feature windows are skipped; a missing future target does
not suppress an observed feature update. Baselines process every available
bar, including session warmup, and missing grid slots advance GARCH by its
conditional expectation. GARCH uses zero-mean standardized t innovations; no
EVT. These are single-instrument baselines, not the enriched portfolio outputs
of the other executables.

**Diurnal profile (shared mode, default on).** Intraday volatility is U-shaped:
the first bars of a session are several times more volatile than midday. At each
calibration the baselines estimate the U factor `s_k` of each slot on the
training window (winsorized mean absolute return, normalized to mean `s^2 = 1`,
the `ewma` module's estimator), run their filters on `r / s_k` and scale the
forecast by the factor of the target bar. This is how `ewma` and `garch_t` treat
intraday returns. Without it, a GARCH-t fitted on raw returns absorbs the U shape
into fat tails (nu about 5 instead of 7) and its midday VaR is too wide: on the
15 synthetic stocks it had 0.57% exceptions at 99% (Kupiec rejected for 9 of 15),
against 0.98% (0 of 15) with the profile. The profile is written to
`diurnal_profile.csv` (last calibration). `--diurnal off` restores raw returns.

**EWMA lambda.** Estimated at each calibration by Gaussian predictive
quasi-likelihood on the training window (first quarter as burn-in, Brent on
[0.80, 0.999]), as in the `ewma` module; `--lambda L` fixes it. Names follow the
repository's definitions: **EWMA-N (ref)** is RiskMetrics (`--diurnal off
--lambda .94`: fixed lambda, raw returns); **EWMA-N** has the estimated lambda
and/or the U profile.

**Opening hour.** CAViaR needs one hour of returns of the same session, so the
first targets of each session start at 11:00 (the 10:00-10:45 bars are never
targets). All models are scored on the same targets, so the comparison is fair,
but it does not cover the opening hour. `report.txt` states the evaluated range
and `backtest_by_hour.csv` breaks the coverage down by hour.

Daily mode fits at the first forecast of each new session, using only labels
whose targets ended before that session. The training window rolls by N
sessions. High-frequency daily boundaries use integer timeline days (86400s)
and explicit gaps; shared mode uses the dataset's sessions. High-frequency N
is the number of timeline sessions touched by the initial training half.
Once mode retains its fitted coefficients while updating filter state.
CAViaR replays observed training features with unobserved labels masked as NaN.
A later failed refit retains the previous filter and records `previous_fit`;
initial fit failure is an error. A non-converged but usable optimizer candidate
is identified in the parameter CSV. No optimization occurs during model updates.

At least 100 observed training labels are required for CAViaR, and at least 100
observed returns **per baseline phase** for comparison. Use longer data or
`--caviar-only` for a shorter high-frequency sample.

## Model and monetary risk

For decimal log returns:

```text
R_t = sum(r_j)
RV_t = sum(r_j * r_j)
q_(t+1) = beta0 + beta1*q_t + beta2*abs(R_t) + beta3*sqrt(RV_t)
```

The recursion advances on every complete feature window. Its target horizon
is the one in the table, independent of update frequency. Estimation minimizes
exact mean pinball loss with deterministic multi-start Nelder-Mead, return
scaling, and a retained empirical constant-quantile candidate. Restrictions
`beta0,beta2,beta3 <= 0` and `0 <= beta1 <= .995` preserve nonpositive quantiles
and stability. They are additional modeling choices, not a global optimality
guarantee. The model implements this repo's explicit realized-variance equation.

Longs fit the lower tail of `r`; shorts fit the lower tail of `-r`, i.e. the
upper asset-return tail. Features are unchanged because they use absolute
returns and squared returns. Forecast CSV quantiles are expressed in the
original asset direction. For signed exposure E and that quantile q:

```text
VaR = -E * expm1(q)
realized P&L = E * expm1(realized log return)
```

The exposure is fixed within a run. It is not a portfolio, fills or cash-account
simulation. Backtests remain in signed-return units, which also permits zero
exposure as a statistical experiment.

## Output contract

| File | Content |
|---|---|
| `forecasts.csv` | one row per model and forecast, original-direction log quantile, features, signed exposure, monetary VaR/P&L, target, exception and evaluation flag |
| `risk_15m.csv` / `risk_5m.csv` | normalized risk rows for the selected horizon; one confidence per run |
| `backtest_summary.csv` | per-model exception rate, Kupiec, Christoffersen independence/CC and mean signed-return pinball |
| `backtest_by_hour.csv` | per model and hour of the target start: n, exceptions, rate, Kupiec p |
| `diurnal_profile.csv` | U factor per slot of the last calibration (shared mode with `--diurnal on`) |
| `params_daily.csv` | calibration effective time, training bounds, model/phase, status, fit time and coefficients (EWMA `lambda` included); includes once-only fits too |
| `benchmark.csv` | mean/p50/p95/p99 ns per forecast, inverse mean latency throughput, persistent model-state bytes, timed updates and batch size |
| `report.txt` | configuration, comparison conventions, metrics and measurement boundaries |

Pending forecasts are emitted with NaN target/P&L, blank exception and
`evaluated=0`. They are never counted as successful observations. Primary
selection depends on target intervals, not future availability. Independence
transitions are counted only for adjacent observed target intervals, excluding
sessions and missing targets. P-values are asymptotic; small 1% tail samples
have low power. No empirical superiority is implied by compilation or tests.

The new CSVs have a **normalized comparison schema**; consumers of the previous
CAViaR `var_long_100000` column must use `exposure` and `var`. They do not copy
all the portfolio-specific columns of the other modules. Timestamps remain
integer seconds: synthetic relative time or exchange-local wall-clock epoch
seconds in shared mode, not necessarily UTC instants.

Latency measures, per forecast, the model update(s) + return quantile +
monetary VaR, timed **in batches of 1000 forecasts** over the out-of-sample
sequence (repeated to at least 2 million per model). A single update takes
10-20 ns, below the 100 ns resolution of the Windows clock, so timing each call
measured the clock: all models reported about 56 ns. Batches spread the clock
overhead to about 0.1 ns per update. The mean is total time / updates; p50/p95/p99
are percentiles of the batch means. Inputs are prepared beforehand, so common
feature construction, lookups, CSV, scoring and fitting are excluded. The
baselines consume all intervening observations of their timeline (including
the division by the U factor). State bytes count the persistent numeric state:
CAViaR coefficients and quantile; EWMA variance and lambda per phase; GARCH
filter and tail factor per phase; the U profile shared by the baselines.
This is model-stage latency, not end-to-end feed latency or a scaling benchmark
of the existing portfolio engines. Fit durations are recorded separately.

## Incremental library

```cpp
#include "rcaviar/data.hpp"
#include "rcaviar/model.hpp"

rcaviar::StreamingWindows features; // fixed 60-return ring; no future prices
rcaviar::Filter filter(fitted.params, fitted.last_quantile);
for (const auto& price : incoming_prices) {
    if (auto w = features.push(price)) {
        const double q = filter.update(w->observation.aggregate_return,
                                       w->observation.realized_variance);
        // Emit q now. Associate a future return with target_time later.
    }
}
// At an explicitly detected session boundary: features.begin_session().
```

Streaming input rejects invalid prices, duplicates, out-of-order timestamps
and non-5s spacing before mutating its state. `begin_session()` resets the
feature window; quantile state is independently owned. The replay adapter
`make_windows()` uses the same builder and attaches labels afterward, including
pending forecasts at the end. `--reset-on-gap` explicitly permits gaps, resets
features and prevents targets crossing them; otherwise gaps fail. Valid extreme
returns are retained.

`rcaviar/replay.hpp` exposes `ReplayInput`, `ReplayConfig`, `run_replay()` and
`write_results()`, so orchestration and reporting can be used independently of
the CLI. `shared_input()` and `price_input()` construct the supported inputs.

## Validation

C++ tests cover recurrence, loss, monetary conversion for longs/shorts,
stream/replay feature equality, pending forecasts, gaps, scaling, daily rolling
fits, failed-fit fallback, common target sets and causal calibration.
Executable-level Python tests independently reconstruct training features,
pinball loss, forecast direction, exposure, phase-specific EWMA recursions and
CSV metrics. They also perturb future prices to check the first forecast.
The shared clean, degraded and B3 datasets are replayed by CTest.

If SciPy is installed, CTest additionally compares the C++ estimator against an
independent linear-programming quantile-regression oracle at fixed persistence
values, on three deterministic conditional-quantile simulations. It permits a
2% objective gap on those fixtures; it does not prove global optimality.

```sh
./realized_caviar/build/rcaviar_fit observations.csv .05
python3 realized_caviar/source/tests/reference/crosscheck_scipy.py \
  realized_caviar/build/rcaviar_fit
cmake -S realized_caviar/source -B realized_caviar/build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug -DRCAVIAR_SANITIZE=ON
cmake --build realized_caviar/build-sanitize --parallel
ctest --test-dir realized_caviar/build-sanitize --output-on-failure
```

`rcaviar_fit` accepts `aggregate_return,realized_variance,target_return` and
returns fitted parameters/diagnostics as JSON. SciPy is only a validation
dependency. No ES, portfolio covariance, options, EVT or live market-data
connector is included; those are outside the minimal scope in DECISIONS.md.
