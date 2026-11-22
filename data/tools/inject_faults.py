"""Builds faulty copies of a dataset, one kind of data problem per copy, with the list of every fault
injected (the truth), to test how the risk engines cope with a bad market-data feed.

Usage:
    python data/tools/inject_faults.py [--data synthetic] [--out data/faults] [--only NAME[,NAME...]]
    python data/tools/inject_faults.py --list

Each scenario is a complete dataset folder, data/faults/<dataset>/<scenario>/, that any engine
reads with --data: a faulty prices_15m.csv, the other files of the dataset unchanged (assets,
portfolio, trades, options) and faults.csv, one row per injected fault:

    fault,timestamp,ticker,detail       timestamp = close time of the affected bar

The faults are drawn with a fixed seed per scenario, so the same command always writes the same
files. The faulty feeds are not versioned (data/faults/ is ignored by git): this script recreates
them. Row order matters: engines that read the file as a feed see the rows in this order (the
arrival order), so out-of-order and late rows are moved in the file. Standard library only.
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
T, TK, O, H, L, C, V = range(7)  # columns of prices_15m.csv


@dataclass
class Feed:
    """The rows of prices_15m.csv in arrival order (lists of strings) and the truth being written."""
    rows: list
    rng: random.Random
    truth: list

    def mark(self, fault, row, detail=""):
        self.truth.append((fault, row[T], row[TK], detail))

    def sessions(self):
        return sorted({r[T][:10] for r in self.rows})

    def tickers(self):
        return sorted({r[TK] for r in self.rows})


def num(x):
    """A price or volume written back with the precision of the source file."""
    return f"{x:.4f}".rstrip("0").rstrip(".") if x == x else ""


def pick(rng, items, k):
    return sorted(rng.sample(list(items), min(k, len(items))))


# ----------------------------------------------------------------------------- faults
# Each fault edits feed.rows in place and records what it did in feed.truth.


def missing(rate):
    def apply(f):
        kept = []
        for r in f.rows:
            if f.rng.random() < rate:
                f.mark("MISSING", r)
            else:
                kept.append(r)
        f.rows[:] = kept
    return apply


def halt(f, episodes=3):
    """A stock does not trade for a whole session (no rows)."""
    hit = {(d, f.rng.choice(f.tickers())) for d in pick(f.rng, f.sessions()[5:], episodes)}
    kept = []
    for r in f.rows:
        if (r[T][:10], r[TK]) in hit:
            f.mark("HALT", r)
        else:
            kept.append(r)
    f.rows[:] = kept


def duplicates(rate):
    def apply(f):
        out = []
        for r in f.rows:
            out.append(r)
            if f.rng.random() < rate:
                out.append(list(r))
                f.mark("DUPLICATE", r)
        f.rows[:] = out
    return apply


def out_of_order(rate, reach=20):
    """A row arrives up to `reach` rows (about a bar) after rows of later bars."""
    def apply(f):
        rows, moved = f.rows, set()
        for i in range(len(rows) - reach):
            if f.rng.random() < rate and id(rows[i]) not in moved:
                j = i + f.rng.randint(1, reach)
                if rows[j][T] > rows[i][T]:
                    f.mark("OUT_OF_ORDER", rows[i], f"after {rows[j][T]}")
                    moved.add(id(rows[i]))
                    rows.insert(j, rows.pop(i))
    return apply


def delay(rate, bars=(2, 4)):
    """A row arrives 2-4 bars late (after the feed has moved past its bar)."""
    def apply(f):
        times = sorted({r[T] for r in f.rows})
        index = {t: k for k, t in enumerate(times)}
        late = []
        kept = []
        for r in f.rows:
            k = index[r[T]] + f.rng.randint(*bars)
            if f.rng.random() < rate and k < len(times) and times[k][:10] == r[T][:10]:
                late.append((times[k], r))
                f.mark("DELAY", r, f"arrives with {times[k]}")
            else:
                kept.append(r)
        for when, r in late:  # after the last row of the bar it arrives with
            at = max(i for i, x in enumerate(kept) if x[T] == when) + 1
            kept.insert(at, r)
        f.rows[:] = kept
    return apply


def stale(f, episodes=3, length=10):
    """A stock's quote freezes: open = high = low = close = last close, volume 0, for `length` bars."""
    by_stock = defaultdict(list)
    for r in f.rows:
        by_stock[r[TK]].append(r)
    for tk in pick(f.rng, f.tickers(), episodes):
        rows = by_stock[tk]
        day = f.rng.choice(f.sessions()[5:])
        idx = [i for i, r in enumerate(rows) if r[T][:10] == day]
        if len(idx) < length + 2:
            continue
        start = idx[1]
        frozen = rows[start - 1][C]
        for r in rows[start:start + length]:
            r[O] = r[H] = r[L] = r[C] = frozen
            r[V] = "0"
            f.mark("STALE", r, f"price frozen at {frozen}")


def bad_ticks(f, count=10, size=(0.05, 0.10)):
    """A single wrong close (5-10% off), the next bar is right again."""
    for r in pick(f.rng, range(len(f.rows)), count):
        r = f.rows[r]
        m = 1 + f.rng.choice((-1, 1)) * f.rng.uniform(*size)
        close = float(r[C]) * m
        r[C] = num(close)
        r[H] = num(max(float(r[H]), close))
        r[L] = num(min(float(r[L]), close))
        f.mark("BAD_TICK", r, f"close x {m:.3f}")


def truncated(f, episodes=3, hours=2):
    """The feed stops `hours` before the close (every stock) on some sessions."""
    for day in pick(f.rng, f.sessions()[5:], episodes):
        last = max(r[T] for r in f.rows if r[T][:10] == day)
        cut = (datetime.strptime(last, "%Y-%m-%d %H:%M:%S") - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        for r in f.rows:
            if r[T][:10] == day and r[T] > cut:
                f.mark("TRUNCATED", r)
        f.rows[:] = [r for r in f.rows if not (r[T][:10] == day and r[T] > cut)]


def clock_shift(f, minutes=60):
    """One session stamped `minutes` late (a time-zone or daylight-saving error in the feed)."""
    day = f.rng.choice(f.sessions()[5:])
    for r in f.rows:
        if r[T][:10] == day:
            f.mark("CLOCK_SHIFT", r, f"stamped {minutes:+d} min")
            r[T] = (datetime.strptime(r[T], "%Y-%m-%d %H:%M:%S") + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def bad_values(rate):
    """Corrupt rows: zero, negative or empty close, or a non-numeric volume."""
    def apply(f):
        for r in f.rows:
            if f.rng.random() < rate:
                kind = f.rng.choice(("zero", "negative", "empty", "text"))
                if kind == "zero":
                    r[C] = "0"
                elif kind == "negative":
                    r[C] = "-" + r[C]
                elif kind == "empty":
                    r[C] = ""
                else:
                    r[V] = "n/a"
                f.mark("BAD_VALUE", r, kind)
    return apply


def unadjusted_split(f):
    """A 2-for-1 split not adjusted in the feed: one stock's prices halve from a session's open on. The
    -50% jump falls in the overnight gap (intraday returns do not see it); the stock's value halves."""
    tk = f.rng.choice(f.tickers())
    sessions = f.sessions()
    day = sessions[len(sessions) // 2]
    first = True
    for r in f.rows:
        if r[TK] == tk and r[T][:10] >= day:
            for k in (O, H, L, C):
                r[k] = num(float(r[k]) / 2)
            r[V] = num(float(r[V]) * 2)
            if first:
                f.mark("UNADJUSTED_SPLIT", r, "prices halved from here on")
                first = False


def late_listing(f, share=0.25):
    """A stock only starts trading after the first quarter of the sessions."""
    tk = f.rng.choice(f.tickers())
    sessions = f.sessions()
    start = sessions[int(len(sessions) * share)]
    for r in f.rows:
        if r[TK] == tk and r[T][:10] < start:
            f.mark("NOT_LISTED", r)
    f.rows[:] = [r for r in f.rows if not (r[TK] == tk and r[T][:10] < start)]


SCENARIOS = {
    "missing_1pct": ([missing(0.01)], "1% of the rows missing at random"),
    "missing_5pct": ([missing(0.05)], "5% of the rows missing at random"),
    "missing_20pct": ([missing(0.20)], "20% of the rows missing at random"),
    "halt": ([halt], "a stock without any row for a whole session (3 times)"),
    "duplicates": ([duplicates(0.02)], "2% of the rows delivered twice"),
    "out_of_order": ([out_of_order(0.01)], "1% of the rows delivered after rows of the next bar"),
    "delay": ([delay(0.005)], "0.5% of the rows delivered 2-4 bars late"),
    "stale": ([stale], "a frozen quote (volume 0) for 10 bars, 3 times"),
    "bad_ticks": ([bad_ticks], "10 single wrong closes, 5-10% off"),
    "truncated": ([truncated], "the feed stops 2 hours before the close on 3 sessions"),
    "clock_shift": ([clock_shift], "one session stamped 1 hour late"),
    "bad_values": ([bad_values(0.005)], "0.5% of the rows with a zero, negative, empty or non-numeric field"),
    "unadjusted_split": ([unadjusted_split], "a 2-for-1 split not adjusted: one stock's prices halve from a session's open on"),
    "late_listing": ([late_listing], "a stock absent from the first quarter of the sessions"),
    "everything": ([missing(0.01), halt, stale, bad_ticks, truncated, bad_values(0.002), duplicates(0.005),
                    out_of_order(0.003), delay(0.002)], "moderate doses of every recoverable fault together"),
}


def build(dataset, name, out_root):
    steps, description = SCENARIOS[name]
    src = os.path.join(DATA, dataset)
    with open(os.path.join(src, "prices_15m.csv"), newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader]
    seed = sum(ord(c) * 31 ** i for i, c in enumerate(f"{dataset}/{name}")) % 2**32
    feed = Feed(rows, random.Random(seed), [])
    for step in steps:
        step(feed)
    out = os.path.join(out_root, dataset, name)
    os.makedirs(out, exist_ok=True)
    for file in os.listdir(src):  # the rest of the dataset, unchanged (not the other price files)
        if file.endswith(".csv") and not file.startswith("prices_15m") and file not in ("scenario.csv", "simulation_truth.csv"):
            shutil.copyfile(os.path.join(src, file), os.path.join(out, file))
    with open(os.path.join(out, "prices_15m.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(feed.rows)
    with open(os.path.join(out, "faults.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["fault", "timestamp", "ticker", "detail"])
        w.writerows(sorted(feed.truth, key=lambda x: (x[1], x[2], x[0])))
    counts = defaultdict(int)
    for fault, *_ in feed.truth:
        counts[fault] += 1
    print(f"  {dataset}/{name:17} {', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}  ({description})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="synthetic,b3_real", help="datasets of data/ to degrade (comma-separated)")
    ap.add_argument("--out", default=os.path.join(DATA, "faults"))
    ap.add_argument("--only", help="scenarios to build (comma-separated; default: all)")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    args = ap.parse_args()
    if args.list:
        for name, (_, description) in SCENARIOS.items():
            print(f"{name:17} {description}")
        return
    names = args.only.split(",") if args.only else list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"unknown scenario(s): {', '.join(unknown)}; see --list")
    for dataset in args.data.split(","):
        for name in names:
            build(dataset, name, args.out)
    print(f"written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
