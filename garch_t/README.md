# garch_t — intraday risk with GARCH-t (VaR and ES every 15 minutes)

Risk engine of the capstone project *Intraday Risk Management in Financial Markets*. It reads the
15-minute prices of a stock portfolio and, **at every 15-minute bar**, computes for the next bar:

- the portfolio **volatility**;
- the **VaR** (the loss exceeded in only 5%, 2.5% or 1% of the bars);
- the **ES** (the average loss when the VaR is exceeded);
- each stock's contribution to the risk.

At the close it also computes the **overnight risk** (up to the end of the first bar of the next
day). If the portfolio holds options, the option book is revalued by Monte Carlo. At the end of a run
it **backtests** every forecast: each one is compared with what actually happened, and statistical
tests say whether the model was right. Everything is compared side by side with the EWMA/RiskMetrics
method described in the project text.

Each stock follows a **GARCH(1,1) model with Student-t innovations**. Around it the engine handles
the intraday seasonality, dynamic correlation between stocks, a portfolio tail based on extreme
value theory, and data with gaps or errors.

---

## Quick start

1. Keep the two folders side by side (this is how they are delivered):

   ```
   ├── garch_t/      this program
   └── data/         the datasets (synthetic/ and b3_real/)
   ```

2. **Windows:** double-click **`run.bat`**. The programs come ready to use in `bin/`: nothing to
   install.
   **Linux / macOS:** run `./build.sh` once, then `./run.sh`.

3. Open the results:
   - `results/synthetic/report.txt` — fictitious market of 15 stocks, 150 trading sessions;
   - `results/b3_real/report.txt` — 15 real B3 stocks, 87 trading sessions.

`run.bat` takes about 30 seconds.

---

## What is in each folder

```
garch_t/
├── README.md            this guide
├── run.bat              Windows: runs the model on both datasets (double-click)
├── run.sh               the same for Linux / macOS
├── build.bat            Windows: rebuilds the programs and runs the automated tests
├── build.sh             the same for Linux / macOS
│
├── bin/                 READY-TO-USE PROGRAMS (Windows 10/11, 64-bit)
│   ├── intraday_risk.exe    the risk engine: VaR/ES every 15 min + backtest   <- the main one
│   ├── monte_carlo.exe      study over many simulated markets (validates the model)
│   ├── generate_data.exe    generates the synthetic market (recreates data/synthetic)
│   ├── fit_series.exe       fits a GARCH to a single series from a CSV file
│   └── benchmark.exe        measures speed
│
├── results/             OUTPUTS: each run writes here, in a folder named after the dataset
│   ├── synthetic/               report.txt + CSV files (fictitious market, with options)
│   ├── synthetic_degraded/      the same market with a faulty data feed
│   ├── b3_real/                 real B3 data
│   ├── monte_carlo_realistic/   study of 60 simulated markets
│   ├── monte_carlo_simple/      study of 30 "well-behaved" markets
│   ├── benchmark.txt            run times
│   └── arch_crosscheck.txt      cross-check against the Python `arch` package
│
└── source/              SOURCE CODE (C++17, no external dependencies)
    ├── CMakeLists.txt       build recipe
    ├── include/garcht/      library interface (.hpp), one file per topic:
    │   ├── garch.hpp            GARCH(1,1)-t model: estimation and filter
    │   ├── portfolio.hpp        portfolio engine (the core of the system)
    │   ├── risk.hpp             VaR and ES (Normal, t, extreme value tail)
    │   ├── backtest.hpp         Kupiec, Christoffersen, Berkowitz and Acerbi–Szekely tests
    │   ├── replay.hpp           the 15-minute cycle over a history
    │   ├── revaluation.hpp      options: Black–Scholes and Monte Carlo
    │   ├── market_data.hpp      CSV reading, dates, holidays, bar grid
    │   ├── seasonality.hpp      volatility pattern along the trading day
    │   ├── synthetic.hpp        generator of the fictitious market
    │   └── math.hpp, optimize.hpp, random.hpp   math, optimizer, random numbers
    ├── src/                 implementation of each .hpp above (.cpp with the same name)
    ├── apps/                the programs in bin/ (one .cpp per program)
    ├── tests/               automated tests (59 tests, 4044 checks)
    └── tools/               fetch_b3_yahoo.py: downloads recent real B3 data
```

The **`data/`** folder (outside this one) has its own `README.md` describing each file.

---

## What the program does, step by step

`intraday_risk` replays the history bar by bar, as if it were running live. At each bar it uses only
the information that existed at that moment.

```
for each 15-minute bar:
  1. measures the P&L of the bar that just closed and compares it with the VaR forecast for it
  2. at the open: records the "gap" (jump between yesterday's close and today's open)
  3. applies the trades (fills) executed in the bar: the portfolio changes
  4. updates each stock's model and the correlation between stocks (microseconds)
  5. at the end of the day: re-estimates all models on the most recent sessions (about 40 ms)
  6. forecasts volatility, VaR and ES for the next bar; with options, runs the book's Monte Carlo
  7. at the close: also forecasts the overnight risk
at the end: backtest of all forecasts + report
```

The first sessions are used only for the first estimation (the **window**: 40 sessions, the same in
every engine of the repository; half of the sessions if the file has fewer). Forecasts start after
it and are always **out of sample**; the window then rolls, one session a day.

### Concepts in one line

| Term | Meaning |
|---|---|
| **VaR 99%** | loss that should be exceeded in only 1% of the bars (for 95%: 5%; for 97.5%: 2.5%) |
| **ES 99%** | average loss in the bars where the 99% VaR was exceeded (measures the size of the hit) |
| **Exception** | a bar where the actual loss exceeded the forecast VaR |
| **GARCH(1,1)** | model where today's volatility depends on yesterday's shock and volatility: turbulent periods cluster |
| **Student t** | distribution with heavier tails than the Normal: large losses are more likely, as in real markets |
| **EWMA-N (ref)** | the reference method of the project text: exponential moving average (λ = 0.94) with a Normal distribution |
| **Backtest** | comparison of the forecasts with what happened; the tests below say whether the error is acceptable |
| **Kupiec** | tests whether the *number* of exceptions matches the expected one (1% at 99%) |
| **Christoffersen (CC)** | also tests whether the exceptions are *spread out* rather than clustered |
| **Acerbi–Szekely (Z2)** | tests whether the ES has the right size |
| **Berkowitz** | tests the *whole* forecast distribution, not just the tail; the most demanding test |
| **Basel traffic light** | regulatory classification by number of exceptions: green (ok), yellow (warning), red (model rejected) |
| **Overnight** | from the close to the end of the first bar of the next day, including the opening jump |

How to read the tests: **p-value above 0.05 = the test found no problem**.

---

## How to use

### 1. With the scripts (easiest)

`run.bat` (or `run.sh`) runs the engine on the synthetic market and on the B3 data, and writes
everything to `results/`.

### 2. From the command line

Open a terminal in the `garch_t` folder (on Windows: right-click the folder → "Open in Terminal"):

```
bin\intraday_risk.exe                              synthetic market (default), with the option book
bin\intraday_risk.exe --data b3_real               real B3 data
bin\intraday_risk.exe --data synthetic --prices ..\data\synthetic\prices_15m_degraded.csv --z-clip 6
                                                   faulty feed; --z-clip 6 also caps the shocks
bin\intraday_risk.exe --data C:\my\folder          your own data (see below)
bin\monte_carlo.exe                                study over 60 simulated markets (about 2 min)
bin\intraday_risk.exe --help                       lists all options
```

On Linux/macOS it is the same, with `./bin/intraday_risk` and forward slashes (`/`).

**Where the data and results are.** The programs find the `data` folder next to `garch_t` by
themselves: `--data synthetic` becomes `data/synthetic`. Results go to
`garch_t/results/<dataset name>`. Both can be changed with `--data FOLDER` and `--out FOLDER`.

### 3. `intraday_risk` options

| Option | What it does | Default |
|---|---|---|
| `--data NAME\|FOLDER` | dataset: `synthetic`, `b3_real` or any folder | `synthetic` |
| `--out FOLDER` | where to write the results | `results/<name>` |
| `--prices FILE` | another price file of the same dataset (e.g. the degraded one) | `prices_15m.csv` |
| `--window-days N` | estimation window, in trading sessions | 40 (half of the sessions if fewer) |
| `--options FILE` / `--no-options` | option book / ignore options | `options.csv`, if present |
| `--mc N` | scenarios of the options Monte Carlo | 10000 |
| `--z-clip Z` | caps shocks at Z standard deviations (protects against bad ticks) | off |
| `--no-jump-filter` (`--jump-z Z`) | feeds the model every print as it is. By default a stock's isolated price jump is held for one bar: dropped if the next bar undoes it (a bad tick), kept one bar late otherwise; jumps of several stocks together are kept at once (see below) | filter on (Z = 10) |
| `--corr ewma\|dcc\|constant` | correlation model between stocks | `ewma` |
| `--corr-lambda L` | fixes the λ of the EWMA correlation instead of estimating it | estimated every day |
| `--dist t\|normal` | distribution of each stock | `t` |
| `--tail-model evt\|t` | portfolio tail: extreme value or plain t | `evt` |
| `--tail oos\|insample` | calibrates the tail on out-of-sample or in-sample outcomes | `oos` |
| `--diurnal per-asset\|pooled`, `--no-diurnal` | intraday pattern per stock, common, or off | `per-asset` |
| `--no-overnight` | skips the overnight risk | computed |
| `--es-sims N` | simulations for the p-value of the ES test (0 = no p-value) | 2000 |
| `--session auto\|HH:MM,MIN,N` | trading session: open, minutes per bar, bars per day | detected |
| `--utc-offset MIN` | time zone used to convert timestamps that carry one (`Z`, `-03:00`) | −180 (São Paulo) |
| `--threads N` | threads for the daily re-estimation | all cores |
| `--daily` | prints a per-day summary | — |
| `--live HOST:PORT` | takes the bars from a live market-data feed as they close (see below) | off (reads the file) |

Invalid values are rejected with an error message, never silently replaced by the default.

**Jump filter (on by default).** A wrong print (one close far off, the next bar back to normal)
enters a GARCH as two extreme shocks, the tick and its reversal, and inflates the variance for many
bars; `--z-clip` only shrinks them. The filter holds a stock's isolated jump (beyond 10 times the
robust scale of its recent returns at that time of day) for one bar: if the next bar undoes it, the
print never enters the model; if the new level holds, the move enters one bar late. Several stocks
jumping together are a market move and enter at once. A jump held at the last bar of a session
never enters the intraday model (the overnight gap starts from the last close kept). Every decision
is written to `jump_events.csv` and counted in the report's `data protection` line.

The filter changes only what the model sees. The backtest scores every forecast on the P&L of the
prices as printed, so the filter cannot hide a loss (an automated test checks that the P&L is
identical with the filter on and off). On the fault catalogue (`atam/results/faults/`):

| Faulty copy of the data | `--no-jump-filter` | `--no-jump-filter --z-clip 6` | filter (default) |
|---|---|---|---|
| b3_real, 10 bad ticks: forecasts off by more than 5% | 77.6% | 85.4% | 0.2% |
| b3_real, all recoverable faults together | 60.0% | 52.6% | 7.3% |
| synthetic, 10 bad ticks | 24.5% | 0.3% | 0.0% |

The cost: a genuine move confined to one stock reaches the VaR one bar late. On the clean b3_real
the filter held 18 jumps in 87 sessions (one a bad tick, 16 genuine moves, one at a session's last
bar); the coverage stays green at every level (99%: 1.15% with the filter, 1.07% without). An
accepted move enters the model at its full size, as the return since the last good close. The
decision on each bar uses only the bars up to it, so it works the same live.

**Live feed.** `--live` connects to a feed server and computes the VaR at each bar as it arrives,
printing one line per bar. `data/tools/feed_server.py` turns any dataset into such a feed (the same
protocol a broker adapter would follow); the dataset's sessions before the first one the feed sends
are the calibration history:

```
python ..\data\tools\feed_server.py --data b3_real --from 2026-11-03 --bar-seconds 1
bin\intraday_risk.exe --data b3_real --live 127.0.0.1:9000       (in a second window)
```

The feed's lines follow the CSV conventions of the files (separators, quotes, decimal commas,
columns in any order). Each bar is closed when the feed's clock passes it; a row that arrives after that is late and is
dropped (counted in the report), a repeated row keeps the first. On a clean feed the live run gives
exactly the same results as the replay of the file (checked by an automated test, and end to end on
`b3_real`). Serve a folder written by `data/tools/inject_faults.py` to see the engine on a faulty
live feed.

### 4. Using your own data

Create a folder with at least two files in the format of `data/` (details in `data/README.md`):

| File | Content | Required |
|---|---|---|
| `assets.csv` | `ticker` column with the portfolio's stocks | yes |
| `prices_15m.csv` | `timestamp,ticker,open,high,low,close,volume`; `timestamp` = **closing** time of the bar | yes |
| `portfolio_initial.csv` | `ticker,quantity` (negative = short) | no (starts empty) |
| `fills.csv` | trades: `timestamp,ticker,side,quantity` (`side` = `BUY`/`SELL`, or empty with a signed quantity) | no |
| `options.csv` | European options | no |

The reader accepts comma or semicolon separators, decimal commas (`38,50`), `YYYY-MM-DD` or
`DD/MM/YYYY` dates and timestamps with a time zone. Rows out of order, duplicated or missing are
accepted too. Then run `bin\intraday_risk.exe --data C:\path\to\folder`.

### 5. The other programs

| Program | Purpose | Example |
|---|---|---|
| `monte_carlo` | runs the model on dozens of independent simulated markets and pools the backtests. With 150 thousand forecasts it shows whether the exception rate is statistically equal to the expected one | `bin\monte_carlo.exe` (realistic) · `bin\monte_carlo.exe --simple --seeds 30` |
| `generate_data` | recreates `data/synthetic` (identical output: the seed is fixed). With `--days`, `--seed` or `--simple` it writes a variant to a folder of its own | `bin\generate_data.exe --seed 7 --out ..\data\variant` |
| `fit_series` | fits a GARCH(1,1) to one column of a CSV and prints parameters, standard errors and VaR as JSON | `bin\fit_series.exe returns.csv --column ret` |
| `benchmark` | measures the speed of estimation, filter and engine | `bin\benchmark.exe > results\benchmark.txt` |

---

## How to read the results

Each run of `intraday_risk` creates a folder in `results/` with a `report.txt` and CSV files (they
open in Excel).

### `report.txt`

**1. Header** — data read, discarded rows (duplicated, invalid) and detected trading session.

**2. Next-bar backtest** — the main block:

```
  model            conf   exc    rate            95% CI expected  Kupiec p     CC p  ES Z2 p    zone ES ratio
  GARCH-t/EVT     0.990    27   0.88% [ 0.60%,  1.27%]    1.00%    0.4854   0.3898   0.6907   green    1.045
  EWMA-N (ref)    0.990    50   1.62% [ 1.23%,  2.14%]    1.00%    0.0014   0.0059   0.0005  yellow    1.247
```

| Column | Meaning |
|---|---|
| `conf` | VaR level (0.990 = 99%) |
| `exc` | number of exceptions (bars where the loss exceeded the VaR) |
| `rate` and `95% CI` | exception rate and its confidence interval; it should contain `expected` |
| `Kupiec p`, `CC p` | p-values of the coverage tests (> 0.05 = ok) |
| `ES Z2 p` | p-value of the ES test (> 0.05 = ok) |
| `zone` | Basel traffic light |
| `ES ratio` | average loss in the exceptions ÷ forecast ES (1 = perfect; above 1 = ES underestimated) |

In the example, GARCH-t gets both VaR and ES right; the project's EWMA-N is yellow and its ES is
rejected (p = 0.0005). The `Berkowitz` line tests the whole distribution (p > 0.05 = ok).

**3. Coverage by time of day** — exception rate at the open (first 3 bars), midday and close (last
3 bars). Shows whether the model fails more at some point of the session.

**4. Other horizons** — `overnight` (from the close to the end of the first bar of the next day)
and, with options, `book MC full reval` (book revalued by Monte Carlo) and `book delta-linear`
(linear approximation, for comparison).

**5. Models at the last re-estimation** — per stock:
- `source`: `fitted` (estimated now), `previous_fit` (estimation failed, uses the previous one) or
  `fallback` (no history yet, uses a conservative variance);
- `alpha`, `beta`: GARCH parameters; `a+b` is the persistence, and the closer to 1, the slower the
  volatility returns to normal;
- `nu`: degrees of freedom of the t (smaller = heavier tails);
- `half-life`: half-life of a shock, in bars;
- `daily vol`: long-run daily volatility.

**6. Last forecast** — σ, 99% VaR and 99% ES (in BRL) for the bar after the end of the data.

**7. Performance** — time per bar and of the daily re-estimation.

### CSV files

| File | One row per… | Main columns |
|---|---|---|
| `risk_15m.csv` | forecast (bar) | `forecast_time` (when it was made), `target_time` (bar forecast), `net_exposure`, `sigma` (std of the P&L, BRL), `var_95/975/99`, `es_95/975/99`, the same measures for the EWMA (`ewma_*`), `realized_pnl` (the P&L that happened) and, with options, `book_mc_var_*`, `book_delta_var_*`, `book_realized_pnl` |
| `risk_overnight.csv` | session | `sigma_gap` (risk of the opening jump), `sigma_first_bar`, `sigma`, `var_*`, `es_*`, `realized_pnl` |
| `backtest_summary.csv` | model × level | everything in the backtest block, plus rates and p-values by time of day |
| `garch_params_daily.csv` | stock × day | parameters estimated each day (`omega`, `alpha`, `beta`, `nu`), robust standard errors (`rse_*`), `source`, `status` |
| `daily_summary.csv` | day | `nu_portfolio`, correlation λ (`corr_b`), re-estimation time (`refit_ms`) |
| `risk_contributions_last.csv` | stock | position, exposure, bar volatility, standalone VaR and **component VaR** (each stock's share of the portfolio VaR; they add up to the total) |
| `correlation_last.csv` | stock | correlation matrix at the last moment |
| `diurnal_profile.csv` | time of day | volatility factor of each bar of the day (the "U") and correlation factor (`correlation_scale`, lower at the auctions) |

`risk_15m.csv` is the main one: plot `var_99` against `realized_pnl` to see the exceptions. The
forecast made at 17:00 refers to the first bar of the next day, without the opening jump, which is
in `risk_overnight.csv`.

In the Monte Carlo studies, `report.txt` and `monte_carlo_summary.csv` give the exception rates
pooled over all markets and the **rejection frequency** of each test. If the model is right, the
tests reject in about 5% of the markets; a wrong model is rejected in almost all of them.

---

## The model

**Each stock.** 15-minute returns, after removing the intraday pattern:

```
y(t) = μ + √h(t) · z(t)          h(t) = ω + α·ε²(t−1) + β·h(t−1)          z ~ standardized Student t
```

- **Maximum likelihood estimation** with an analytic gradient and the BFGS optimizer. Each day it
  starts from the previous day's parameters. Classic and robust standard errors. Same conventions as
  the Python `arch` package, which the results match.
- **Low-priced stocks** (many zero returns because of the tick size) and degenerate fits are
  re-estimated automatically in a more robust way.
- **A stock without history** (new listing, dead feed) gets a conservative variance until there is
  data.
- **Missing bar:** the variance moves forward by its expected value; a return covering k bars is
  divided by √k.

**Intraday pattern.** Volatility has a U shape along the day (high at the open and at the close).
Each stock has its own profile, pulled towards the common profile unless the data show a real
difference (empirical Bayes).

**Portfolio.**
- **Correlation** between stocks by EWMA of the standardized shocks, with λ **estimated every day**.
  Alternatives: DCC(1,1) or constant correlation.
- **Auctions:** correlation in the first and last bar of the day is lower (auctions have
  stock-specific moves). A factor between 0 and 1 per auction is estimated and only applied when
  there is evidence. On B3: about 0.6 at the open and 0.3 at the close.
- **A halted stock** during the session: the correlation keeps updating with the other stocks,
  without freezing the rest.
- **Portfolio tail** by extreme value theory (McNeil & Frey): the body is a t, and the 10% largest
  losses follow a generalized Pareto distribution. It is calibrated on out-of-sample outcomes once
  there are 500 of them.
- **Component VaR** per stock (Euler decomposition).

**Overnight.** Variance of each opening jump by EWMA, a correlation of its own for the jumps, and a
tail calibrated on the current portfolio. It is added to the first bar of the next day.

**Options.** Monte Carlo with 10 thousand scenarios from the joint distribution forecast by the
model, repricing each option with Black–Scholes. Time to expiry counts B3 business days. The
delta-linear approximation is computed alongside, for comparison.

**Project reference.** EWMA covariance of the returns (λ = 0.94) with a Normal distribution,
computed in parallel for comparison.

---

## Results

**Simulated markets** (`results/monte_carlo_realistic`; 60 markets, 185 thousand forecasts per model):

| Model | 95% VaR exceptions | 97.5% VaR | 99% VaR | ES test rejections at 99% | Berkowitz rejections |
|---|---|---|---|---|---|
| **GARCH-t** | 5.12% | 2.50% | 1.02% | 5% (as expected for a correct model) | 13% |
| EWMA-N (ref) | 5.01% | 3.09% | 1.81% | 100% | 100% |

At the open, EWMA-N reaches 5.2% of exceptions at 99%; GARCH-t stays at 1.2%.

**Real B3 data** (`results/b3_real`; 87 sessions, 1,308 out-of-sample forecasts):

| Model | 95% VaR | 97.5% VaR | 99% VaR | ES test (p) | Traffic light |
|---|---|---|---|---|---|
| **GARCH-t** | 4.66% | 2.14% | 1.15% | 0.26 | green at all three levels |
| EWMA-N (ref) | 4.97% | 3.13% | 1.68% | 0.0005 (rejected) | yellow at 99% |

The Berkowitz test of the whole distribution is at the limit for GARCH-t (p = 0.045; 0.06 with
`--no-jump-filter`) and rejects EWMA-N clearly (p = 0.001).

**Synthetic market** (`results/synthetic`): GARCH-t 5.17% / 2.37% / 0.88%, Berkowitz p = 0.63.
With the degraded feed (and `--z-clip 6`): 5.20% / 2.44% / 0.84%, with no run-time failure.
The Monte Carlo study measures the model alone, on markets with no faulty prints, so it runs with
the jump filter off (the library default).

---

## Tests and validation

`build.bat` (or `build.sh`) rebuilds and runs **59 automated tests (4044 checks)** and 7 end-to-end
tests. They cover:

- **Math functions:** compared with SciPy and with 40-digit reference values (mpmath).
- **Estimation:** matches the Python `arch` package (`results/arch_crosscheck.txt`, produced with
  `python source/tests/reference/crosscheck_arch.py bin/fit_series.exe`); recovers the true
  parameters of simulated series.
- **Statistical tests:** compared with reference values.
- **Options:** Black–Scholes against known values and put-call parity.
- **Robustness:** faulty data, halted stock, new listing, assorted CSV formats and time zones,
  several threads at once, folders with accented names.
- **Live feed:** the same market fed row by row, with a session cut short, gives exactly the
  forecasts of the file replay, with the jump filter off and on; late and repeated rows are counted
  and dropped.
- **Jump filter:** a bad tick (also at a session's first bar), a genuine shift, a move at a
  session's last bar and a market move are each classified as expected; the decisions are causal;
  the scored P&L is the same with the filter on and off.

It compiles with no warnings on GCC, Clang and Visual Studio, for Windows, Linux and macOS.

**Performance** (15 stocks, 8-core computer):

| Operation | Time |
|---|---|
| Update at each bar (trades, filters, correlation, VaR/ES) | about 4 µs |
| Full daily re-estimation | about 40 ms |
| Monte Carlo of the option book (10 thousand scenarios) | about 6.5 ms per bar |
| Portfolio of 200 stocks, per bar | about 160 µs |

---

## Using it as a C++ library

```cpp
#include "garcht/garcht.hpp"
using namespace garcht;

IntradayRiskEngine engine(tickers, EngineConfig{});         // default configuration (described above)
engine.on_overnight(opening_gaps);                          // once per session, at the open (optional)
engine.on_bar(slot, bar_returns);                           // every 15 min; NaN = no trade
engine.observe_outcome(realized_pnl, previous_forecast);    // feeds the tail calibration
if (end_of_day) engine.refit(exposures_in_brl);             // daily re-estimation
PortfolioRisk r = engine.compute(next_slot, exposures_in_brl);   // r.sigma, r.garch[k].var, r.garch[k].es
OvernightRisk o = engine.compute_overnight(exposures_in_brl);    // at the close
```

Several threads can query the engine at the same time while another one feeds it. The complete
cycle, ready to use, is `run_replay()` (`source/include/garcht/replay.hpp`). To build it into another
project, add `source/` to its CMake (`add_subdirectory`) and link to the `garcht` library.

---

## Known limitations

- **Short real history.** Yahoo keeps only ~60 sessions of 15-minute bars; `fetch_b3_yahoo.py`
  appends new sessions, and `b3_real` has 87. On these data VaR and ES pass at every level; the
  Berkowitz test of the whole distribution is at the limit (p = 0.045); a longer B3 history would
  make the validation firmer.
- **Live feed and trading hours.** The live mode takes the session grid from the history, the file
  replay from the whole file: if the hours change inside the file (see below), give `--session`.
- **B3 trading hours.** The close moves between 17:00 and 18:00 with US daylight saving time. For a
  period crossing the change, run each part separately with `--session`.
- **B3 holidays** are registered for 2025–2027 (function `is_b3_holiday` in
  `source/src/market_data.cpp`).
- **Options:** Black–Scholes with a fixed implied volatility per contract, no volatility surface.
- **Data quality:** the project's NORMAL / DEGRADED / FREEZE modes belong to the data-quality layer.
  The engine provides its inputs: source and status of each model, inactive stocks, unscored
  forecasts.

---

## Troubleshooting

| Message | Solution |
|---|---|
| `dataset 'synthetic' not found` | the `data` folder must be next to `garch_t`; or give its path with `--data` |
| `bin\intraday_risk.exe not found` | run `build.bat` (Windows) or `./build.sh` (Linux/macOS) |
| `Permission denied` / `cannot write` when saving | an antivirus may block freshly built programs: allow the program or delete the old results folder |
| Windows SmartScreen warns when opening an `.exe` | the programs are not digitally signed: "More info" → "Run anyway" |
| `CMake not found` when building | install WinLibs: `winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --source winget` |
