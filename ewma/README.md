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
   - `results/b3_real/` — 15 real B3 stocks, 60 sessions.

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
    │   ├── engine.hpp           portfolio engine: on_bar / refit / compute / compute_overnight
    │   ├── data_quality.hpp     DQ rules, dq_score, modes, detection against the truth
    │   ├── backtest.hpp         Kupiec, Christoffersen, Wilson, traffic light, Berkowitz, Acerbi–Szekely
    │   ├── replay.hpp           the 15-minute cycle over a history + backtest of all forecasts
    │   └── ewmar.hpp            includes everything
    ├── src/                 implementation of each .hpp (.cpp with the same name)
    ├── apps/                the programs in bin/
    └── tests/               automated tests + reference/scipy_reference.py (SciPy values)
```

The implementation is independent of `garch_t`: no code is shared, only the `data/` folder.

---

## What the program does, step by step

`ewma_risk` replays the history bar by bar, as if it were running live, using at each bar only
what was known at that moment:

```
rows of the price file, in file order (= the order they arrived from the feed)
  └─ data-quality layer: duplicates, late rows, gaps, frozen quotes -> the bars
for each 15-minute bar:
  1. decides the suspicious prices of the previous bar (bad tick or real shock?)
  2. scores the previous forecast: P&L of the bar vs VaR (overnight gap excluded)
  3. applies the trades (fills) of the bar
  4. checks the new prices for jumps (|z| > 6), computes dq_score and the mode
  5. updates the three EWMA filters (microseconds); FREEZE: no update
  6. at the end of the day: re-estimates the U profile, λ and ν on the window (a few ms)
  7. forecasts σ, VaR and ES of the next bar for the three models; at the close, the overnight
at the end: backtest of all forecasts + report
```

The window, the first forecast and the scoring are the same as in `garch_t`: the first sessions
(half of them, at most 60) are used only for estimation, and every forecast is **out of sample**.

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
| `--window-days N` | estimation window, in sessions | half of the sessions, 20 to 60 |
| `--lambda L` | fixes the λ of EWMA-N / EWMA-t | estimated every day |
| `--nu N` | fixes the ν of EWMA-t | estimated every day |
| `--diurnal pooled\|per-asset\|off` | U profile common to all stocks, one per stock, or none | `pooled` |
| `--no-overnight` | skips the overnight risk | computed |
| `--options FILE` / `--no-options` | option book / ignore options | `options.csv`, if present |
| `--no-dq` | turns the data-quality layer off (plain loader, as in `garch_t`) | on |
| `--dq-thresholds A,B` | dq_score limits of NORMAL and FREEZE | `0.9,0.6` |
| `--truth FILE\|none` | clean price file for the detection rates and the robustness comparison | `prices_15m.csv` of the dataset |
| `--es-sims N` | simulations for the p-value of the ES test | 2000 |
| `--session auto\|HH:MM,MIN,N` | session: open, minutes per bar, bars per day | detected |
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
| `risk_15m.csv` | forecast | `ref_*` (EWMA-N (ref)), `sigma`, `lambda`, `nu`, `n_*` (EWMA-N), `t_*` (EWMA-t) VaR/ES at the three levels, `realized_pnl`, `dq_score`, `mode`, `frozen`; with options, `book_*` |
| `backtest_summary.csv` | model × level | **the same columns as `garch_t`** |
| `risk_overnight.csv` | session | `sigma_gap`, `sigma_first_bar`, `sigma`, VaR/ES t (`var_*`) and Normal (`n_var_*`), realized P&L |
| `daily_summary.csv` | day | λ, ν, overnight λ and ν, refit time |
| `diurnal_profile.csv` | time of day | the U factor of each bar |
| `risk_contributions_last.csv` | stock | position, exposure, volatility, standalone and component VaR 99% (EWMA-t) |
| `correlation_last.csv` | stock | correlation matrix at the end |
| `dq_events.csv` | event | time, stock, flag, detail |
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
| `SPIKE` | \|z\| > 6 against the forecast σ: the EWMA is updated with the value capped at ±6σ. If the next bar takes the price back (less than 25% of the move left), it was a bad tick: the price is excluded and the bar re-fed. If not, the shock is real and is re-fed in full (`SHOCK_KEPT`). If several stocks jump at the same bar, it is a market move and is kept at once | — |

`dq_score` = Σ wᵢ qᵢ, wᵢ = |exposure|ᵢ / Σ|exposure|, qᵢ = 1 minus the penalties of the stock's flags
(GAP, STALE, SPIKE: 1; OUT_OF_ORDER: 0.1; DUPLICATE: 0.05). Modes, with hysteresis (a better mode
only after 3 bars that qualify for it):

| Mode | dq_score | What happens |
|---|---|---|
| NORMAL | ≥ 0.9 | normal operation |
| DEGRADED | 0.6 to 0.9 | forecasts published with a mark; `warning` alert |
| FREEZE | < 0.6 | the EWMA is not updated, the last valid VaR/ES is kept; `critical` alert |

---

## Results

### Synthetic market (`results/synthetic`, 2517 out-of-sample forecasts)

| Model | 95% VaR | 97.5% VaR | 99% VaR | zone 99% | ES test p (99%) | ES ratio (99%) | Berkowitz p |
|---|---|---|---|---|---|---|---|
| EWMA-N (ref) | 4.69% | 2.46% | 1.35% | yellow | 0.0005 | 1.31 | 0.002 |
| EWMA-N | 4.65% | 2.50% | 1.51% | yellow | 0.0005 | 1.17 | 0.095 |
| **EWMA-t** | 5.13% | 2.22% | 1.19% | **green** | **0.17** | **1.04** | **0.45** |
| GARCH-t/EVT (`garch_t`) | 5.16% | 2.42% | 1.03% | green | 0.29 | 1.08 | 0.76 |

- EWMA-N (ref) reproduces the `garch_t` reference exactly: 118 / 62 / 34 exceptions.
- Removing the U profile fixes the open: at 99%, exceptions in the first 3 bars fall from 4.1%
  (ref) to 2.6% (EWMA-N) and 1.9% (EWMA-t).
- The t tail fixes the ES: its test passes and the ES ratio goes from 1.17 to 1.04.

### Real B3 data (`results/b3_real`, 840 forecasts)

| Model | 95% VaR | 97.5% VaR | 99% VaR | zone 99% | ES test p (99%) | ES ratio (99%) |
|---|---|---|---|---|---|---|
| EWMA-N (ref) | 4.29% | 3.33% | 2.02% | yellow | 0.0005 | 1.32 |
| EWMA-N | 4.76% | 2.98% | 1.43% | green | 0.013 | 1.28 |
| **EWMA-t** | 5.00% | 2.86% | 1.31% | green | **0.083** | **1.18** |
| GARCH-t/EVT (`garch_t`) | 5.60% | 2.74% | 1.07% | green | 0.21 | 1.18 |

EWMA-N (ref) again equals the `garch_t` reference (36 / 28 / 17). At the open, EWMA-N (ref) has
12.2% of exceptions at 99%; EWMA-N and EWMA-t, 2.2%.

### Data-quality experiment (same market, three runs)

Detection on the faulty feed (`results/synthetic_degraded/dq_detection.csv`):

| Fault | Injected | Detected |
|---|---|---|
| missing rows (`GAP`) | 285 | 285 (100%) |
| duplicates | 171 | 171 (100%) |
| out-of-order rows | 129 | 126 accepted as `OUT_OF_ORDER`; 3 arrived after their bar closed (`DELAY`) |
| frozen XCON3, day 130 (`STALE`) | 10 bars | 10 (2 flagged after the fact, already used) |
| bad ticks XPET3, XBAN4, XTEC3 (`SPIKE`) | 3 | 3 (100%) |

No false bad tick on the clean file: the −9σ crash and the −9% shock are kept (a market move and a
shock not reverted). FREEZE is never triggered: no bar loses 40% of the exposure's data.

| Run (EWMA-t) | 95% / 97.5% / 99% exceptions | VaR 99 vs clean run: mean / max deviation | bars off by > 1% |
|---|---|---|---|
| clean feed | 129 / 56 / 30 | — | — |
| faulty feed, DQ on | 127 / 56 / 29 | 0.48% / 8.0% | 267 |
| faulty feed, DQ off | 123 / 57 / 27 | 2.74% / 51.0% | 980 |

Without the layer, each bad tick inflates the variance for hours (VaR up to 51% too high); with it,
the largest difference is 8%, at the bar where the bad tick was still undecided.

---

## Tests and validation

`build.bat` (or `build.sh`) rebuilds and runs **27 unit tests (415 checks)** and 4 end-to-end
replays (clean, faulty with and without DQ, B3). They cover:

- **Math:** Φ⁻¹, t distribution (cdf, pdf, quantile), incomplete beta and χ² against SciPy, deep in
  the tails.
- **VaR/ES factors** of the t against SciPy numerical integration.
- **Estimation:** recovers the U profile and λ of simulated series, and ν of simulated t samples.
- **EWMA:** the covariance stays positive semidefinite with 15% of missing data; undoing a bar
  restores the filters exactly.
- **Backtest:** Kupiec, Christoffersen and Wilson against SciPy; Basel zones; Berkowitz and ES test
  accept a correct model and reject a wrong one.
- **Data quality:** each rule on a small series with its fault injected; score, modes and
  hysteresis; bad tick vs real jump inside a replay.
- **Data:** CSV formats, decimal comma, time zones, B3 calendar, return conventions.

**Performance** (`results/benchmark.txt`):

| Stocks | Update per bar (median) | Daily re-estimation |
|---|---|---|
| 15 | under 1 µs | about 4 ms |
| 50 | about 3 µs | about 12 ms |
| 200 | about 60 µs | about 85 ms |

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
engine.compute(next_slot, exposures_in_brl, r);    // r.ref / r.normal / r.student: sigma, VaR, ES
OvernightRisk o = engine.compute_overnight(exposures_in_brl);
```

The complete cycle with the data-quality layer is `run_replay()` (`source/include/ewmar/replay.hpp`).

---

## Known limitations

- **Short real history**: 60 sessions of B3 (Yahoo limit) give 840 forecasts; the Berkowitz test
  still rejects EWMA-N on them, and EWMA-t is at the edge (p = 0.05).
- **One λ for all stocks** (needed to keep the covariance positive semidefinite).
- **Overnight**: gap and first bar assumed independent.
- **Bad ticks at the close**: a jump at the last bar is decided with the next day's first bar, so the
  overnight move enters the decision. On B3 one closing print (GGBR4, 2026-09-14) is excluded this way.
- **SPIKE** needs the next bar: during that bar the update uses the value capped at ±6σ.
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
