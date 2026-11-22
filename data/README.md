# Data — 15-minute intraday bars

Two datasets in the same format, ready to test any intraday risk model (VaR, ES, volatility). The
files are plain CSV (comma separator, decimal point) and open in Excel, Python (pandas) or any other
tool.

```
data/
├── synthetic/    fictitious market: 15 stocks, 150 sessions, with a portfolio, trades and options
├── b3_real/      real B3 data: 15 stocks, 87 sessions
└── tools/        inject_faults.py: faulty copies of the datasets, to test robustness
```

## `synthetic/` — fictitious market

- **15 fictitious stocks** (XPET3, XELE3, XBAN4, …) in 5 sectors, **150 trading sessions** from
  2026-03-02 to 2026-10-02, **28 bars of 15 minutes per session** (10:00–17:00), B3 business days
  only.
- Simulated with a fixed seed: the `generate_data` program of `garch_t` recreates exactly the same
  files.
- The generating process imitates a real market and is deliberately *different* from a plain GARCH:
  - market factor, sector factors and a stock-specific component;
  - leverage effect (falls raise volatility more than rises);
  - calm/agitated volatility regimes;
  - U-shaped pattern along the day, different for each stock, and jumps at the opening and closing
    auctions;
  - three low-priced stocks where the BRL 0.01 tick matters;
  - scripted events (listed in `scenario.csv`): a −9 standard-deviation crash with a circuit breaker
    and a −9% shock in one stock.
- **Portfolio:** initial position (`portfolio_initial.csv`), 21 trades over the period (`fills.csv`)
  and 4 European options (`options.csv`).
- **`prices_15m_degraded.csv`:** the same market with a faulty data feed (missing bars, duplicated
  and out-of-order rows, a frozen price, bad ticks), to test robustness.
- **`simulation_truth.csv`:** the true parameters of the generating process, to check whether an
  estimator recovers what was simulated.

## `b3_real/` — real B3 data

- **15 stocks:** PETR4, PRIO3, VALE3, GGBR4, SUZB3, ITUB4, BBDC4, BBAS3, B3SA3, ABEV3, RADL3,
  RENT3, WEGE3, SBSP3 and EQTL3.
- **87 trading sessions** from 2026-07-17 to 2026-11-19, 15-minute bars (Yahoo Finance), São Paulo
  local time.
- **Real imperfections are kept as they are:** bars without trades have no row; 2026-07-31 has bars
  only from 13:00 on, and 2026-10-26 and 2026-11-16 only from 11:00 on.
- **Portfolio:** about BRL 700 thousand in each stock and a short position in B3SA3. No trades and
  no options.
- **To add more recent sessions:** `python garch_t/source/tools/fetch_b3_yahoo.py`. Yahoo keeps only
  the last ~60 sessions of 15-minute bars, so the script keeps every session already in the file and
  appends the new ones; run it at least every two months and the history keeps growing.

## Faulty copies — testing robustness

A real market-data feed fails in many ways. `tools/inject_faults.py` writes one faulty copy of a
dataset per kind of problem, so each engine can be run on it and compared with its run on the clean
data:

```
python data/tools/inject_faults.py            # every scenario, for synthetic and b3_real
python data/tools/inject_faults.py --list     # what each scenario does
python data/tools/inject_faults.py --data b3_real --only bad_ticks,stale
```

Each copy is a complete dataset folder, `data/faults/<dataset>/<scenario>/`, that any engine reads
with `--data` (for example `garch_t\bin\intraday_risk.exe --data data\faults\b3_real\stale`). Next
to the faulty `prices_15m.csv` it holds `faults.csv`, the list of every fault injected (`fault`,
`timestamp` of the affected bar, `ticker`, `detail`), so a data-quality check can be scored against
the truth.

| Scenario | What goes wrong |
|---|---|
| `missing_1pct`, `missing_5pct`, `missing_20pct` | 1%, 5% or 20% of the rows never arrive |
| `halt` | a stock has no row for a whole session (3 times) |
| `duplicates` | 2% of the rows arrive twice |
| `out_of_order` | 1% of the rows arrive after rows of the next bar |
| `delay` | 0.5% of the rows arrive 2–4 bars late |
| `stale` | a quote freezes (same price, volume 0) for 10 bars (3 times) |
| `bad_ticks` | 10 single wrong closes, 5–10% off; the next bar is right again |
| `truncated` | the feed stops 2 hours before the close (3 sessions) |
| `clock_shift` | one session stamped one hour late (a time-zone error) |
| `bad_values` | 0.5% of the rows with a zero, negative, empty or non-numeric field |
| `unadjusted_split` | a 2-for-1 split the feed did not adjust: one stock's prices halve from a session's open on (a fake −50% overnight gap; the position's value halves) |
| `late_listing` | a stock only starts trading after the first quarter of the sessions |
| `everything` | moderate doses of every recoverable fault together |

The faults are drawn with a fixed seed, so the same command always writes the same files; the
copies are not versioned (they take about 85 MB) and the script recreates them in a few seconds.
The rows are in **arrival order**: an engine that reads the file as a feed sees late and
out-of-order rows where they arrived. The ATAM evaluation runs every engine on every copy
(`atam/results/faults/`).

## File formats

**`prices_15m.csv`** — one row per stock per bar:

| Column | Meaning |
|---|---|
| `timestamp` | **closing** time of the bar, `YYYY-MM-DD HH:MM:SS`, São Paulo local time (the 10:00–10:15 bar is stamped `10:15:00`) |
| `ticker` | stock code |
| `open`, `high`, `low`, `close` | open, high, low and close prices of the bar (BRL) |
| `volume` | number of shares traded in the bar |

A bar without trades has no row.

**`assets.csv`** — `ticker`, `name`, `sector`, `market_beta` (true beta in the synthetic market;
empty for B3).

**`portfolio_initial.csv`** — `ticker`, `quantity`: position at the start of the period, in shares
(negative = short).

**`fills.csv`** — executed trades: `fill_id`, `timestamp` (time of the trade), `ticker`, `side`
(`BUY` / `SELL`), `quantity` (shares), `price` (BRL).

**`options.csv`** — European options:

| Column | Meaning |
|---|---|
| `id` | identifier |
| `underlying` | underlying stock |
| `type` | `CALL` or `PUT` |
| `strike` | strike price (BRL) |
| `expiry` | expiry date, `YYYY-MM-DD` (at the close) |
| `implied_vol` | annual implied volatility (0.32 = 32%) |
| `rate` | annual interest rate, continuously compounded (0.14 = 14%) |
| `multiplier` | shares per contract |
| `quantity` | contracts (negative = short) |

**`scenario.csv`** — `day`, `timestamp`, `event`: scripted events of the synthetic market.

**`simulation_truth.csv`** — true parameters of each process (market, sectors and each stock):
`daily_vol` (daily volatility), `alpha`, `gamma_leverage`, `beta` (GJR-GARCH dynamics), `nu`
(degrees of freedom of the t), `market_beta`, `sector_loading` and `diurnal_bend` (strength of the
intraday U shape).
