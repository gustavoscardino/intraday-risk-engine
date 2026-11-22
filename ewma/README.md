# ewma — intraday risk with EWMA (VaR and ES every 15 minutes)

EWMA module of the capstone project *Intraday Risk Management in Financial Markets*. It reads the
15-minute prices of a stock portfolio and, **at every 15-minute bar**, forecasts the risk of the
next bar with **three EWMA models side by side**:

| Model | Volatility | Distribution | What it isolates |
|---|---|---|---|
| **EWMA-N (ref)** | RiskMetrics: λ = 0.94 fixed, raw returns | Normal | the classic method of the project text (same as the "EWMA-N (ref)" of `garch_t`) |
| **EWMA-N** | λ estimated every day, intraday pattern removed | Normal | the gain from a better **volatility** |
| **EWMA-t** | the same σ as EWMA-N | Student t, ν estimated every day | the gain from a fatter **tail** |

Going ref → EWMA-N → EWMA-t → GARCH-t (the `garch_t` module) splits the improvement into three
steps: volatility dynamics, tail, and model.

The module also has the **data-quality layer** of the project specification: it checks the price
feed (duplicated, late, missing, frozen and wrong prices), scores each bar and switches the engine
between the **NORMAL / DEGRADED / FREEZE** modes. At the close it forecasts the **overnight risk**;
with options, the risk of the option book (delta-linear). Every run ends with a full **backtest**,
with the same tests and the same output columns as `garch_t`, so the two modules can be compared
line by line.

---

## Quick start

1. Keep the folders side by side (this is how the project is organized):

   ```
   ├── data/       the datasets (synthetic/ and b3_real/)
   ├── garch_t/    the GARCH-t module
   └── ewma/       this module
   ```

2. **Windows:** double-click **`run.bat`**. The programs come ready to use in `bin/`.
   **Linux / macOS:** run `./build.sh` once, then `./run.sh`.

3. Open the results, starting with `report.txt` in each folder:
   - `results/synthetic/` — fictitious market of 15 stocks, 150 sessions, clean feed;
   - `results/synthetic_degraded/` — the same market with a faulty feed, data-quality layer on;
   - `results/synthetic_degraded_nodq/` — the faulty feed with the layer off;
   - `results/b3_real/` — 15 real B3 stocks, 87 sessions.

`run.bat` takes a few seconds.

---

## What is in each folder

```
ewma/
├── README.md            this guide
├── run.bat / run.sh     run the four scenarios
├── build.bat / build.sh rebuild the programs and run the automated tests
│
├── bin/                 READY-TO-USE PROGRAMS (Windows 10/11, 64-bit)
│   ├── ewma_risk.exe        the risk engine: VaR/ES every 15 min + DQ layer + backtest
│   └── benchmark.exe        measures speed (15, 50 and 200 stocks)
│
├── results/             OUTPUTS: one folder per scenario + benchmark.txt
│
└── source/              SOURCE CODE (C++17, no external dependencies)
    ├── CMakeLists.txt       build recipe (library `ewmar` + programs + tests)
    ├── include/ewmar/       library interface, one file per topic:
    │   ├── math.hpp             Normal and Student t, incomplete beta/gamma, Brent, random numbers
    │   ├── market_data.hpp      CSV, timestamps, B3 calendar, session grid, panel, returns
    │   ├── seasonality.hpp      intraday volatility pattern (the "U")
    │   ├── ewma.hpp             multivariate EWMA filter, starting covariance, λ estimation
    │   ├── risk.hpp             VaR/ES factors (Normal and t), ν estimation, Black–Scholes
    │   ├── models.hpp           the three models behind the common model interface (common/)
    │   ├── engine.hpp           portfolio engine: drives the models, window, overnight risk
    │   ├── data_quality.hpp     DQ rules, dq_score, modes, detection against the truth
    │   ├── backtest.hpp         Kupiec, Christoffersen, Wilson, traffic light, Berkowitz, Acerbi–Szekely
    │   ├── replay.hpp           the 15-minute cycle over a history + backtest of all forecasts
    │   └── ewmar.hpp            includes everything
    ├── src/                 implementation of each .hpp (.cpp with the same name)
    ├── apps/                the programs in bin/
    └── tests/               automated tests + reference/scipy_reference.py (SciPy values)
```

The engine is built on its own (namespace `ewmar`, its own build and programs) and does not link
`garch_t`; some infrastructure was adapted from it (the `paths.hpp` / `cli.hpp` helpers, the UTF-8
manifest, the B3 holiday table and parts of the CSV reader), and both read the same `data/` folder.
Both compile the shared library `common/`: the common model interface, the RiskMetrics baseline
(EWMA-N (ref) is the same code in both engines) and the jump filter.

---

## What the program does, step by step

`ewma_risk` replays the history bar by bar, as if it were running live, using at each bar only
what was known at that moment:

```
rows of the price file, in file order (= the order they arrived from the feed)
  └─ data-quality layer: duplicates, late rows, gaps, frozen quotes -> the bars
for each 15-minute bar:
  1. scores the previous forecast: P&L of the bar vs VaR (overnight gap excluded)
  2. applies the trades (fills) of the bar
  3. the jump filter decides the bar's prints (an isolated jump is held until the next bar),
     computes dq_score and the mode
  4. updates the three models (microseconds); FREEZE: no update
  5. at the end of the day: re-estimates the U profile, λ and ν on the window (a few ms)
  6. forecasts σ, VaR and ES of the next bar for the three models; at the close, the overnight
at the end: backtest of all forecasts + report
```

The window, the first forecast and the scoring are the same as in `garch_t`: the first sessions
(40, the same window in every engine; half of the sessions if the file has fewer) are used only for
estimation, and every forecast is **out of sample**.

### Concepts in one line

| Term | Meaning |
|---|---|
| **EWMA** | exponentially weighted moving average of squared returns: σ²(t) = λ σ²(t−1) + (1−λ) r²(t). λ close to 1 = long memory |
| **λ estimated** | the λ that best predicted the variance of the last window (maximum quasi-likelihood), re-estimated every day |
| **U profile** | volatility is higher at the open and at the close; returns are divided by the factor of their time of day |
| **Student t** | distribution with heavier tails than the Normal; ν (degrees of freedom) measures how heavy |
| **VaR 99%** | loss exceeded in only 1% of the bars |
| **ES 99%** | average loss in the bars where the 99% VaR is exceeded |
| **Exception** | a bar where the actual loss exceeded the forecast VaR |
| **dq_score** | quality of the bar's data, weighted by the exposure in each stock (1 = perfect) |

How to read the tests: **p-value above 0.05 = the test found no problem**.

---

## How to use

### Command line

Open a terminal in the `ewma` folder:

```
bin\ewma_risk.exe                                   synthetic market (default)
bin\ewma_risk.exe --data b3_real                    real B3 data
bin\ewma_risk.exe --data synthetic --prices ..\data\synthetic\prices_15m_degraded.csv
                                                    faulty feed (results/synthetic_degraded)
bin\ewma_risk.exe --data synthetic --prices ..\data\synthetic\prices_15m_degraded.csv --no-dq
                                                    the same without the DQ layer (results/synthetic_degraded_nodq)
bin\ewma_risk.exe --help                            all options
```

On Linux/macOS: `./bin/ewma_risk` and forward slashes.

### `ewma_risk` options

| Option | What it does | Default |
|---|---|---|
| `--data NAME\|FOLDER` | dataset: `synthetic`, `b3_real` or any folder in the `data/` format | `synthetic` |
| `--prices FILE` | another price file of the dataset (e.g. the degraded one) | `prices_15m.csv` |
| `--out FOLDER` | where to write the results | `results/<name>[_degraded][_nodq]` |
| `--window-days N` | estimation window, in sessions | 40 (half of the sessions if fewer) |
| `--lambda L` | fixes the λ of EWMA-N / EWMA-t | estimated every day |
| `--nu N` | fixes the ν of EWMA-t | estimated every day |
| `--diurnal pooled\|per-asset\|off` | U profile common to all stocks, one per stock, or none | `pooled` |
| `--no-overnight` | skips the overnight risk | computed |
| `--options FILE` / `--no-options` | option book / ignore options | `options.csv`, if present |
| `--no-dq` | turns the data-quality layer off (plain loader, no jump filter) | on |
| `--no-jump-filter` | keeps the layer but lets every isolated jump into the models | filter on |
| `--dq-thresholds A,B` | dq_score limits of NORMAL and FREEZE | `0.9,0.6` |
| `--truth FILE\|none` | clean price file for the detection rates and the robustness comparison | `prices_15m.csv` of the dataset |
| `--es-sims N` | simulations for the p-value of the ES test | 2000 |
| `--session auto\|HH:MM,MIN,N` | session: open, minutes per bar, bars per day | detected |
| `--resample` | the price file has samples finer than the `--session` grid (e.g. 5-second prices): each bar takes the first open, the high, the low, the last close and the summed volume of its samples | off |
| `--utc-offset MIN` | time zone for timestamps that carry one | −180 (São Paulo) |
| `--daily` | prints λ, ν and refit time per day | — |

Invalid values are rejected with a message, never replaced by the default.

---

## How to read the results

### `report.txt`

1. **Header**: data read, session, model settings.
2. **Backtest** of the three models at 95%, 97.5% and 99% (same columns as `garch_t`): exceptions,
   rate and 95% interval, Kupiec and Christoffersen p-values, ES test (Acerbi–Szekely Z2), Basel
   traffic light, ES ratio (1 = perfect; above 1 = ES underestimated), and the Berkowitz test of the
   whole distribution.
3. **Coverage by time of day**: open (first 3 bars), midday, close (last 3 bars).
4. **Other horizons**: overnight (Normal and t) and, with options, the delta-linear book.
5. **EWMA parameters**: λ and ν of the last day and their range over the period.
6. **Data-quality layer**: rows discarded, flagged cells, bad ticks, time in each mode, alerts;
   **detection against the clean file** and, for a faulty feed, the **robustness** comparison with
   the same market through a clean feed.
7. **Last forecast** and **performance**.

### CSV files

| File | One row per… | Content |
|---|---|---|
| `risk_15m.csv` | forecast | one block per model (common/README.md): `ref_*` (EWMA-N (ref): sigma, VaR/ES at the three levels), `n_*` (EWMA-N, with `n_lambda`), `t_*` (EWMA-t, with `t_nu`); `realized_pnl`, `dq_score`, `mode`, `frozen`; with options, `book_*` |
| `backtest_summary.csv` | model × level | **the same columns as `garch_t`** |
| `risk_overnight.csv` | session | `sigma_gap`, `sigma_first_bar`, `sigma`, VaR/ES t (`var_*`) and Normal (`n_var_*`), realized P&L |
| `daily_summary.csv` | day | λ, ν, overnight λ and ν, refit time |
| `diurnal_profile.csv` | time of day | the U factor of each bar |
| `risk_contributions_last.csv` | stock | position, exposure, volatility, standalone and component VaR 99% (EWMA-t) |
| `correlation_last.csv` | stock | correlation matrix at the end |
| `dq_events.csv` | event | time, stock, flag, detail |
| `jump_events.csv` | decision of the jump filter | time, stock, action (`HELD`, `BAD_TICK`, `SHOCK_ACCEPTED`, `MARKET_MOVE`, `UNDECIDED`), return, z (the same file as `garch_t`) |
| `alerts.csv` | alert | time, severity (`info` / `warning` / `critical`), type, message |
| `dq_detection.csv` | flag | faults injected, detected, recall, flags on other cells |

---

## The model

**Returns.** Same conventions as `garch_t`: the first bar of a session is ln(close/open), so the
overnight gap is excluded; a return covering k bars (missing bars) is divided by √k.

**Intraday pattern.** s_k = mean |r| at time of day k (winsorized at the 99.5% quantile),
normalized so that the mean of s² is 1; y = r / s_k. Common to all stocks (`pooled`) or one per
stock (`--diurnal per-asset`). Re-estimated every day on the window.

**Multivariate EWMA.** Σ(t) = λ Σ(t−1) + (1−λ) y(t) y(t)ᵀ, one λ for all stocks (the matrix stays
positive semidefinite). It starts from the sample covariance of the first quarter of the window,
which is not scored. **Missing data:** pairs of observed stocks are updated; a missing stock keeps
its variance, and its covariances with the observed ones are multiplied by √λ. This keeps Σ
positive semidefinite (Σ_new = D Σ D + (1−λ) y yᵀ); simply keeping the old covariances, as first
planned, can break it.

**λ every day.** Maximizes the Gaussian predictive quasi-likelihood
Σᵢ Σₜ [−½ ln σ²ᵢ(t|t−1) − y²ᵢ(t) / (2σ²ᵢ(t|t−1))] over the window, by Brent on [0.80, 0.999].

**Forecast.** a_i = x_i · s_{k(t+1)} (x = exposures in BRL), σ = √(aᵀΣa).
VaR = σ·q, ES = σ·e, with
- Normal: q = Φ⁻¹(c), e = φ(q)/(1−c);
- t with unit variance: x = t_ν⁻¹(1−c), k = √((ν−2)/ν), q = −k·x, e = k·f_ν(x)/(1−c)·(ν+x²)/(ν−1).

**ν every day.** Maximum likelihood of a unit-variance t (Brent on [2.5, 60]) fitted to the
standardized one-step-ahead portfolio outcomes of the window, xᵀr(t) / σ(t|t−1), with the current
exposures.

**Overnight.** An EWMA covariance of the opening gaps per session, with its own λ (estimated the
same way once there are 40 sessions); ν fitted on the standardized portfolio gaps. The gap is added
to the first bar of the next day, assumed independent.

**Options.** Delta-linear only: the Black–Scholes delta of each option (B3 business days to
expiry) is added to the exposure of its underlying; the book's realized P&L is computed with full
Black–Scholes revaluation.

### Data-quality layer

The order of the rows in the price file is taken as the order in which they arrived from the feed.
Each stock has its own watermark (newest bar received). Rules:

| Flag | Rule | Action |
|---|---|---|
| `DUPLICATE` | same (stock, bar) already received | row discarded |
| `OUT_OF_ORDER` | bar older than the stock's watermark, but its bar is still open | row accepted |
| `DELAY` | the bar already closed (the feed is more than 1 bar past it) | row discarded |
| `GAP` | other stocks printed the bar, this one did not | missing return |
| `STALE` | close unchanged **and** volume 0 for ≥ 3 bars in a row (volume 0 separates a frozen quote from a cheap stock that did not move) | price ignored from the 3rd bar on |
| `SUSPECT` | the jump filter shared by the three engines (`common/`): a return beyond 10 times the stock's robust scale (median absolute return of its last 10 sessions at that time of day), while fewer than 3 stocks jump together, is held: the models see the bar as not printed | price held |
| `SPIKE` | the next bar of the stock undid more than half of the held jump: a bad tick | price never enters the models |
| `SHOCK_KEPT` | the new level held (the move enters one bar late, at full size), or several stocks jumped together (a market move, kept at once) | price enters |

`dq_score` = Σ wᵢ qᵢ, wᵢ = |exposure|ᵢ / Σ|exposure|, qᵢ = 1 minus the penalties of the stock's flags
(GAP, STALE, SUSPECT, SPIKE: 1; OUT_OF_ORDER: 0.1; DUPLICATE: 0.05). Modes, with hysteresis (a better mode
only after 3 bars that qualify for it):

| Mode | dq_score | What happens |
|---|---|---|
| NORMAL | ≥ 0.9 | normal operation |
| DEGRADED | 0.6 to 0.9 | forecasts published with a mark; `warning` alert |
| FREEZE | < 0.6 | the EWMA is not updated, the last valid VaR/ES is kept; `critical` alert |

---

## Results

### Synthetic market (`results/synthetic`, 3077 out-of-sample forecasts)

| Model | 95% VaR | 97.5% VaR | 99% VaR | zone 99% | ES test p (99%) | ES ratio (99%) | Berkowitz p |
|---|---|---|---|---|---|---|---|
| EWMA-N (ref) | 4.61% | 2.66% | 1.62% | yellow | 0.0005 | 1.25 | 0.0004 |
| EWMA-N | 4.71% | 2.57% | 1.66% | yellow | 0.0005 | 1.14 | 0.050 |
| **EWMA-t** | 4.87% | 2.47% | 1.20% | **green** | **0.16** | **1.02** | **0.26** |
| GARCH-t/EVT (`garch_t`) | 5.17% | 2.37% | 0.88% | green | 0.68 | 1.05 | 0.67 |

- EWMA-N (ref) is the same code as the `garch_t` reference (`common/`) and gives the same
  forecasts on the same inputs: 142 / 82 / 50 exceptions in both.
- Removing the U profile fixes the open: at 99%, exceptions in the first 3 bars fall from 5.2%
  (ref) to 2.7% (EWMA-N) and 1.8% (EWMA-t).
- The t tail fixes the ES: its test passes and the ES ratio goes from 1.14 to 1.02.

### Real B3 data (`results/b3_real`, 87 sessions, 1308 forecasts)

| Model | 95% VaR | 97.5% VaR | 99% VaR | zone 99% | ES test p (99%) | ES ratio (99%) |
|---|---|---|---|---|---|---|
| EWMA-N (ref) | 4.97% | 3.13% | 1.68% | yellow | 0.0005 | 1.32 |
| EWMA-N | 5.28% | 3.13% | 1.91% | yellow | 0.0005 | 1.23 |
| **EWMA-t** | 5.66% | 2.83% | 1.38% | green | 0.033 | **1.18** |
| GARCH-t/EVT (`garch_t`) | 4.74% | 2.22% | 1.15% | green | 0.24 | 0.99 |

EWMA-N (ref) again equals the `garch_t` reference (65 / 41 / 22). At the open, EWMA-N (ref) has
6.7% of exceptions at 99%; EWMA-N and EWMA-t, 1.5%.

### Data-quality experiment (same market, three runs)

Detection on the faulty feed (`results/synthetic_degraded/dq_detection.csv`):

| Fault | Injected | Detected |
|---|---|---|
| missing rows (`GAP`) | 285 | 285 (100%) |
| duplicates | 171 | 171 (100%) |
| out-of-order rows | 126 | 126 (100%), accepted as `OUT_OF_ORDER` |
| late rows, arrived after their bar closed (`DELAY`) | 3 | 3 (100%), discarded |
| frozen XCON3, day 130 (`STALE`) | 10 bars | 10 (2 flagged after the fact, already used) |
| bad ticks XPET3, XBAN4, XTEC3 (`SPIKE`, the shared jump filter) | 3 | 3 (100%) |

The clean file has no arrival order, so the injected out-of-order and late rows are taken from the
order in which the faulty file delivers them, with the same rule the layer applies: those two lines
check that the rule runs as specified, not that it finds every fault. The independent measure, against
the list of every fault written by `data/tools/inject_faults.py`, is in `atam/results/faults/`.

No false bad tick on the clean file: the −9σ crash and the −9% shock are kept (a market move and a
shock not reverted). FREEZE is never triggered: no bar loses 40% of the exposure's data.

| Run (EWMA-t) | 95% / 97.5% / 99% exceptions | VaR 99 vs clean run: mean / max deviation | bars off by > 1% |
|---|---|---|---|
| clean feed | 150 / 76 / 37 | — | — |
| faulty feed, DQ on | 153 / 77 / 40 | 0.52% / 5.0% | 365 |
| faulty feed, DQ off | 146 / 79 / 39 | 3.02% / 48.6% | 1447 |

Without the layer, each bad tick inflates the variance for hours (VaR up to 49% too high); with it,
the bad ticks never reach the models (the exceptions are counted on the prices as received, so the
faulty run scores the bad ticks' own bars too). On the faulty copies of `atam/` (every recoverable
fault together, B3 data) 6.3% of the EWMA-N forecasts move more than 5%, against 50.4% without the
layer.

---

## Tests and validation

`build.bat` (or `build.sh`) rebuilds and runs **29 unit tests (429 checks)**, the tests of the shared library (`common/`) and 4 end-to-end
replays (clean, faulty with and without DQ, B3). They cover:

- **Math:** Φ⁻¹, t distribution (cdf, pdf, quantile), incomplete beta and χ² against SciPy, deep in
  the tails.
- **VaR/ES factors** of the t against SciPy numerical integration.
- **Estimation:** recovers the U profile and λ of simulated series, and ν of simulated t samples.
- **EWMA:** the covariance stays positive semidefinite with 15% of missing data.
- **Model interface:** the three models are registered with their names and column prefixes; the
  EWMA-N (ref) forecast is that of the shared RiskMetrics model on the same window.
- **Backtest:** Kupiec, Christoffersen and Wilson against SciPy; Basel zones; Berkowitz and ES test
  accept a correct model and reject a wrong one.
- **Data quality:** each rule on a small series with its fault injected; score, modes and
  hysteresis; bad tick vs real jump inside a replay: the bad tick leaves the VaR where the clean
  feed puts it, and the P&L the forecasts are scored on is the same with the filter on and off.
- **Other grids:** 5-second samples aggregate into 1-minute bars; 15-minute rows split into 5-minute
  samples and resampled give the same rows and the same replay.
- **Data:** CSV formats, decimal comma, time zones, B3 calendar, return conventions.
- **Replay:** a feed cut inside a session leaves every earlier forecast, overnight one included,
  as in the whole replay.

**Performance** (`results/benchmark.txt`, 40-session window; timings vary by up to a factor of two
between sessions on the same laptop):

| Stocks | Update per bar (median) | Daily re-estimation (median) |
|---|---|---|
| 15 | about 1 µs | about 6 ms |
| 50 | about 7 µs | about 18 ms |
| 200 | about 130 µs | about 140 ms |

---

## Using it as a C++ library

```cpp
#include "ewmar/ewmar.hpp"
using namespace ewmar;

EwmaEngine engine(n_assets, EngineConfig{});
engine.on_overnight(opening_gaps);                 // at the open (optional)
engine.on_bar(slot, bar_returns);                  // every 15 min; NaN = no trade
if (end_of_day) engine.refit(exposures_in_brl);    // daily re-estimation
PortfolioRisk r;
engine.compute(next_slot, exposures_in_brl, r);    // r.models[k]: sigma, VaR, ES of engine.models()[k]
OvernightRisk o = engine.compute_overnight(exposures_in_brl);
```

The complete cycle with the data-quality layer is `run_replay()` (`source/include/ewmar/replay.hpp`).

---

## Known limitations

- **Short real history**: 87 sessions of B3 give 1308 forecasts. On them the Berkowitz test rejects
  the three EWMA models (EWMA-t p = 0.010) and the ES test rejects EWMA-t at 99% (p = 0.033), while
  GARCH-t passes both; a longer history would make the comparison firmer.
- **One λ for all stocks** (needed to keep the covariance positive semidefinite).
- **Overnight**: gap and first bar assumed independent.
- **Jumps at the close**: a jump held at the last bar of a session is undecided and never enters the
  intraday models (the overnight gap starts from the last close kept).
- **The jump filter** needs the next bar: a genuine move confined to one stock reaches the VaR one
  bar late.
- **Horizon**: one bar; a 5-minute VaR every minute on 5-second data is five runs on 5-minute grids
  opening one minute apart (`--resample --session 10:00,5,84`, `10:01,5,83`, ...).
- **Options**: delta-linear only (no gamma); fixed implied volatility per contract.
- **B3 holidays** registered for 2025–2027 (`is_b3_holiday` in `source/src/market_data.cpp`).

---

## Troubleshooting

| Message | Solution |
|---|---|
| `dataset 'synthetic' not found` | the `data` folder must be next to `ewma`, or give its path with `--data` |
| `bin\ewma_risk.exe not found` | run `build.bat` (Windows) or `./build.sh` |
| `CMake not found` when building | install WinLibs: `winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --source winget` |
| build errors about C++17 | an old compiler (e.g. `C:\MinGW`, GCC 6) is being used: `build.bat` puts WinLibs first in the PATH; delete `source\build` and run again |
| Windows SmartScreen warns about an `.exe` | the programs are not signed: "More info" → "Run anyway" |
