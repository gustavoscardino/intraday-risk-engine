"""Downloads real 15-minute bars of B3 stocks (Yahoo Finance, last ~60 days) and writes
them in the engine's CSV format.

Usage:
    python fetch_b3_yahoo.py [--out DIR] [--range 60d] [--notional 700000] [--replace]

DIR defaults to the dataset folder next to garch_t (data/b3_real).

Yahoo keeps only the last ~60 sessions of 15-minute bars, so by default the dataset GROWS: the
sessions already in DIR/prices_15m.csv are kept as they are and only the sessions after its last
day are appended (assets.csv and portfolio_initial.csv are kept too). Today's session is always
left out (it may be incomplete); run the script again the next day to add it. --replace rebuilds the
folder from the download alone.

Writes DIR/assets.csv, DIR/prices_15m.csv (bar *close* times, Sao Paulo local time),
DIR/portfolio_initial.csv (roughly equal notional, one short hedge) and an empty
DIR/fills.csv. Only the Python standard library is used.

Timestamps from Yahoo are bar *start* times in UTC; they are converted to the exchange's local
time and shifted to the close.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

TICKERS = [
    ("PETR4", "Petrobras preferred", "Energy"),
    ("PRIO3", "PRIO common", "Energy"),
    ("VALE3", "Vale common", "Materials"),
    ("GGBR4", "Gerdau preferred", "Materials"),
    ("SUZB3", "Suzano common", "Materials"),
    ("ITUB4", "Itau Unibanco preferred", "Financials"),
    ("BBDC4", "Bradesco preferred", "Financials"),
    ("BBAS3", "Banco do Brasil common", "Financials"),
    ("B3SA3", "B3 common", "Financials"),
    ("ABEV3", "Ambev common", "Consumer"),
    ("RADL3", "Raia Drogasil common", "Consumer"),
    ("RENT3", "Localiza common", "Industrials"),
    ("WEGE3", "WEG common", "Industrials"),
    ("SBSP3", "Sabesp common", "Utilities"),
    ("EQTL3", "Equatorial common", "Utilities"),
]
SHORT_HEDGE = "B3SA3"  # one short position to exercise signed exposures
BAR_MINUTES = 15


def fetch(symbol, rng):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.SA"
           f"?interval={BAR_MINUTES}m&range={rng}&includePrePost=false")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # unknown / delisted symbol: no point in retrying
                print(f"warning: {symbol}: not found on Yahoo, skipped", file=sys.stderr)
                return None
            if attempt == 3:
                print(f"warning: {symbol}: {exc}, skipped", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))
        except Exception as exc:  # network hiccups / rate limits: retry with backoff
            if attempt == 3:
                print(f"warning: {symbol}: {exc}, skipped", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


def to_rows(symbol, payload):
    result = payload["chart"]["result"][0]
    offset = timedelta(seconds=result["meta"].get("gmtoffset", -10800))
    quote = result["indicators"]["quote"][0]
    now = datetime.now(timezone.utc)
    rows = []
    for i, ts in enumerate(result.get("timestamp") or []):
        o, h, l, c, v = (quote[k][i] for k in ("open", "high", "low", "close", "volume"))
        if c is None:  # Yahoo leaves gaps as nulls: keep them missing
            continue
        start_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        close_utc = start_utc + timedelta(minutes=BAR_MINUTES)
        if close_utc > now:  # bar still forming
            continue
        local = (close_utc + offset).replace(tzinfo=None)
        rows.append((local.strftime("%Y-%m-%d %H:%M:%S"), symbol,
                     o if o is not None else c, h if h is not None else c,
                     l if l is not None else c, c, int(v or 0)))
    # Yahoo appends the closing-auction print as a flat zero-volume point after the last regular
    # bar (e.g. a "17:15" bar on the day of the download): drop such trailing points per day.
    by_day = {}
    for r in rows:
        by_day.setdefault(r[0][:10], []).append(r)
    out = []
    for day in sorted(by_day):
        rs = by_day[day]
        while rs and rs[-1][6] == 0 and rs[-1][2] == rs[-1][3] == rs[-1][4] == rs[-1][5]:
            rs.pop()
        out.extend(rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    # default: data/b3_real next to garch_t (this file is garch_t/source/tools/fetch_b3_yahoo.py)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "..", "..", "data", "b3_real"))
    ap.add_argument("--range", default="60d")
    ap.add_argument("--notional", type=float, default=700000.0)
    ap.add_argument("--replace", action="store_true", help="rebuild from the download alone instead of appending")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prices_path = os.path.join(args.out, "prices_15m.csv")
    existing = []
    if not args.replace and os.path.exists(prices_path):
        with open(prices_path, newline="") as f:
            existing = [tuple(r) for r in csv.reader(f)][1:]

    with ThreadPoolExecutor(max_workers=5) as pool:
        payloads = list(pool.map(lambda t: fetch(t[0], args.range), TICKERS))

    today = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%d")  # Sao Paulo date
    all_rows, last_price = [], {}
    for (sym, _, _), payload in zip(TICKERS, payloads):
        rows = [r for r in (to_rows(sym, payload) if payload else []) if r[0][:10] < today]
        if not rows:
            print(f"warning: no data for {sym}", file=sys.stderr)
            continue
        all_rows.extend(rows)
        first_day = rows[0][0][:10]
        last_price[sym] = next(r[5] for r in rows if r[0][:10] == first_day)
    if not all_rows and not existing:
        sys.exit("error: no bars were downloaded (network, or Yahoo returned no data); nothing written")
    all_rows.sort(key=lambda r: (r[0], r[1]))
    formatted = [(ts, sym, f"{o:.4f}", f"{h:.4f}", f"{l:.4f}", f"{c:.4f}", str(v)) for ts, sym, o, h, l, c, v in all_rows]

    if existing:  # keep every session already in the file; append only the later sessions
        last_day = max(r[0][:10] for r in existing)
        added = [r for r in formatted if r[0][:10] > last_day]
        new_days = sorted({r[0][:10] for r in added})
        print(f"kept {len({r[0][:10] for r in existing})} sessions up to {last_day}; "
              f"appended {len(new_days)} sessions" + (f" ({new_days[0]} .. {new_days[-1]})" if new_days else ""))
        formatted = sorted(existing + added, key=lambda r: (r[0], r[1]))
    with open(prices_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "ticker", "open", "high", "low", "close", "volume"])
        w.writerows(formatted)
    days = sorted({r[0][:10] for r in formatted})
    print(f"{len(formatted)} bars, {len(days)} sessions ({days[0]} .. {days[-1]}) -> {os.path.abspath(args.out)}")
    if existing and os.path.exists(os.path.join(args.out, "assets.csv")) and os.path.exists(os.path.join(args.out, "portfolio_initial.csv")):
        return  # the stocks and the initial portfolio stay as they were
    with open(os.path.join(args.out, "assets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "name", "sector", "market_beta"])
        for sym, name, sector in TICKERS:
            if sym in last_price:
                w.writerow([sym, name, sector, ""])
    with open(os.path.join(args.out, "portfolio_initial.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "quantity"])
        for sym, _, _ in TICKERS:
            if sym not in last_price:
                continue
            qty = round(args.notional / last_price[sym] / 100) * 100
            w.writerow([sym, -qty // 2 if sym == SHORT_HEDGE else qty])
    with open(os.path.join(args.out, "fills.csv"), "w", newline="") as f:
        f.write("fill_id,timestamp,ticker,side,quantity,price\n")


if __name__ == "__main__":
    main()
