# Data — 15-minute intraday bars

Two datasets in the same format, ready to test any intraday risk model (VaR, ES, volatility). The
files are plain CSV (comma separator, decimal point) and open in Excel, Python (pandas) or any other
tool.

```
data/
├── synthetic/    fictitious market: 15 stocks, 150 sessions, with a portfolio, trades and options
└── b3_real/      real B3 data: 15 stocks, 87 sessions
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
