"""Matched comparison over every stock of the shared datasets, with Diebold-Mariano tests.

Runs `realized_caviar` once per stock of data/synthetic and data/b3_real (Realized-CAViaR, EWMA-N,
GARCH-t and EWMA-t on the same targets), with a FIXED estimation window passed explicitly (the
default would depend on the length of the file), and consolidates:

  summary.csv          per dataset and model: exceptions pooled over the stocks, Kupiec / CC
                       rejections, mean pinball loss, stocks where the model has the lowest loss,
                       model-stage latency and state size
  by_stock.csv         the backtest of every stock and model
  by_hour.csv          exceptions per hour of the target start, pooled over the stocks
  diebold_mariano.csv  every pair of models: per stock and pooled, the mean pinball-loss
                       difference, the DM statistic (Newey-West variance) and its p-value
  report.txt           the same as readable tables, with the configuration of the run

Raw runs go to realized_caviar/results/comparison/ (not versioned); the consolidated tables to
realized_caviar/comparison/ (versioned). Standard library only.

    python realized_caviar/tools/compare_all.py [--confidence .99] [--jobs 4]
"""
import argparse
import concurrent.futures
import csv
import itertools
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
REPO = MODULE.parent
WINDOWS = {"synthetic": 60, "b3_real": 30}  # sessions; synthetic has 150, B3 60 (half, at most 60)
ORDER = ["Realized-CAViaR", "EWMA-N", "EWMA-t", "GARCH-t"]


def find_exe():
    for rel in ("build/realized_caviar.exe", "build/Release/realized_caviar.exe", "build/realized_caviar"):
        p = MODULE / rel
        if p.exists():
            return p
    sys.exit("realized_caviar is not built: run realized_caviar/build.bat (Windows) or build.sh first")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def diebold_mariano(d):
    """DM test of mean(d) = 0 with a Newey-West long-run variance (Bartlett weights).

    d: loss differentials in time order. Returns (n, mean, stat, two-sided p). Targets are
    one step ahead and non-overlapping, so the lag only guards against residual autocorrelation."""
    n = len(d)
    if n < 10:
        return n, float("nan"), float("nan"), float("nan")
    mean = sum(d) / n
    c = [x - mean for x in d]
    lags = int(4 * (n / 100) ** (2 / 9))
    lrv = sum(x * x for x in c) / n
    for k in range(1, lags + 1):
        gamma = sum(c[t] * c[t - k] for t in range(k, n)) / n
        lrv += 2 * (1 - k / (lags + 1)) * gamma
    if not lrv > 0:
        return n, mean, float("nan"), float("nan")
    stat = mean / math.sqrt(lrv / n)
    return n, mean, stat, 2 * norm_sf(abs(stat))


def run_stock(exe, dataset, ticker, confidence, raw_root):
    out = raw_root / dataset / ticker
    cmd = [str(exe), "--data", str(REPO / "data" / dataset), "--ticker", ticker,
           "--window-days", str(WINDOWS[dataset]), "--confidence", str(confidence), "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{dataset} {ticker}: {proc.stderr.strip()}")
    return dataset, ticker, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", type=Path, default=MODULE / "comparison")
    ap.add_argument("--datasets", default="synthetic,b3_real")
    args = ap.parse_args()
    tau = 1 - args.confidence
    exe = find_exe()
    raw_root = MODULE / "results" / "comparison"
    datasets = [d for d in args.datasets.split(",") if d]
    # code version, read before anything is written (the output folder itself is not code)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "realized_caviar", "ewma/source", "garch_t/source",
                            f":(exclude){args.out.resolve().relative_to(REPO).as_posix()}" if args.out.resolve().is_relative_to(REPO) else "."],
                           capture_output=True, text=True).stdout.strip()

    jobs = [(ds, row["ticker"]) for ds in datasets for row in read(REPO / "data" / ds / "assets.csv")]
    print(f"{len(jobs)} runs ({', '.join(f'{d}: window {WINDOWS[d]} sessions' for d in datasets)}), {args.jobs} at a time")
    runs = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for ds, ticker, out in pool.map(lambda j: run_stock(exe, j[0], j[1], args.confidence, raw_root), jobs):
            runs.append((ds, ticker, out))
            print(f"  {ds:9} {ticker}")

    by_stock, by_hour, losses, latency = [], defaultdict(lambda: [0, 0]), {}, defaultdict(list)
    for ds, ticker, out in runs:
        for r in read(out / "backtest_summary.csv"):
            by_stock.append({"dataset": ds, "ticker": ticker, **r})
        for r in read(out / "backtest_by_hour.csv"):
            acc = by_hour[(ds, r["model"], int(r["hour"]))]
            acc[0] += int(r["n"]); acc[1] += int(r["exceptions"])
        for r in read(out / "benchmark.csv"):
            latency[(ds, r["model"])].append((float(r["mean_ns"]), int(r["state_bytes"])))
        # per-forecast pinball loss of the evaluated targets (the runs use a positive exposure)
        series = defaultdict(dict)
        for r in read(out / "forecasts.csv"):
            if r["evaluated"] != "1":
                continue
            y, q = float(r["target_return"]), float(r["quantile"])
            series[r["model"]][int(r["target_time"])] = (y - q) * (tau - (y < q))
        losses[(ds, ticker)] = series
        for r in read(out / "backtest_summary.csv"):  # sanity: same loss as the engine reports
            s = series[r["model"]]
            assert math.isclose(sum(s.values()) / len(s), float(r["mean_pinball"]), rel_tol=1e-9), (ds, ticker, r["model"])

    models = [m for m in ORDER if any(r["model"] == m for r in by_stock)]
    models += sorted({r["model"] for r in by_stock} - set(models))

    # ---- summary per dataset and model
    summary = []
    for ds in datasets:
        rows = [r for r in by_stock if r["dataset"] == ds]
        best = defaultdict(int)
        for ticker in {r["ticker"] for r in rows}:
            stock = [r for r in rows if r["ticker"] == ticker]
            best[min(stock, key=lambda r: float(r["mean_pinball"]))["model"]] += 1
        for m in models:
            mr = [r for r in rows if r["model"] == m]
            n = sum(int(r["n"]) for r in mr); exc = sum(int(r["exceptions"]) for r in mr)
            lat = sorted(x[0] for x in latency[(ds, m)])
            summary.append({
                "dataset": ds, "model": m, "stocks": len(mr), "n": n, "exceptions": exc,
                "rate": exc / n, "expected_rate": tau,
                "kupiec_rejections": sum(float(r["kupiec_p"]) < 0.05 for r in mr),
                "cc_rejections": sum(float(r["conditional_coverage_p"]) < 0.05 for r in mr),
                "mean_pinball": sum(float(r["mean_pinball"]) for r in mr) / len(mr),
                "lowest_pinball_stocks": best[m],
                "median_latency_ns": lat[len(lat) // 2],
                "state_bytes": latency[(ds, m)][0][1],
            })

    # ---- Diebold-Mariano on the pinball loss: per stock and pooled (cross-sectional mean)
    dm_rows = []
    for ds in datasets:
        tickers = sorted(t for d, t in losses if d == ds)
        for a, b in itertools.combinations(models, 2):
            pooled = defaultdict(list)
            for t in tickers:
                la, lb = losses[(ds, t)][a], losses[(ds, t)][b]
                times = sorted(set(la) & set(lb))
                d = [la[x] - lb[x] for x in times]
                for x in times:
                    pooled[x].append(la[x] - lb[x])
                n, mean, stat, p = diebold_mariano(d)
                dm_rows.append({"dataset": ds, "scope": t, "model_a": a, "model_b": b, "n": n,
                                "mean_loss_diff": mean, "dm_stat": stat, "p_value": p})
            d = [sum(v) / len(v) for _, v in sorted(pooled.items())]
            n, mean, stat, p = diebold_mariano(d)
            dm_rows.append({"dataset": ds, "scope": "pooled", "model_a": a, "model_b": b, "n": n,
                            "mean_loss_diff": mean, "dm_stat": stat, "p_value": p})

    # ---- outputs
    args.out.mkdir(parents=True, exist_ok=True)

    def write(name, rows, fields):
        with open(args.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    write("summary.csv", summary, list(summary[0]))
    write("by_stock.csv", by_stock, ["dataset", "ticker"] + [k for k in by_stock[0] if k not in ("dataset", "ticker")])
    write("by_hour.csv", [{"dataset": ds, "model": m, "hour": h, "n": v[0], "exceptions": v[1], "rate": v[1] / v[0]}
                          for (ds, m, h), v in sorted(by_hour.items())], ["dataset", "model", "hour", "n", "exceptions", "rate"])
    write("diebold_mariano.csv", dm_rows, list(dm_rows[0]))

    lines = [
        "Matched comparison: Realized-CAViaR, EWMA-N, EWMA-t and GARCH-t on every stock",
        f"VaR confidence {args.confidence:g} (expected exception rate {tau:.2%}); daily rolling calibration;",
        "estimation window fixed per dataset: " + ", ".join(f"{d} {WINDOWS[d]} sessions" for d in datasets) + ".",
        f"Code: commit {commit}{' with uncommitted changes' if dirty else ''}; machine: {platform.processor() or platform.machine()}.",
        "Targets: 15-minute bars from 11:00 to 17:00 (CAViaR needs one hour of the session); every model on the same targets.",
        "",
    ]
    for ds in datasets:
        lines.append(f"{ds}: {sum(1 for t in losses if t[0] == ds)} stocks")
        lines.append(f"  {'model':16} {'n':>6} {'exc':>5} {'rate':>7} {'Kupiec rej':>11} {'CC rej':>7} {'pinball (bp)':>13} {'lowest loss':>12} {'ns/update':>10}")
        for s in (x for x in summary if x["dataset"] == ds):
            lines.append(f"  {s['model']:16} {s['n']:6d} {s['exceptions']:5d} {s['rate']:7.2%} {s['kupiec_rejections']:5d}/{s['stocks']:<5d}"
                         f" {s['cc_rejections']:3d}/{s['stocks']:<3d} {1e4 * s['mean_pinball']:13.4f} {s['lowest_pinball_stocks']:6d}/{s['stocks']:<5d}"
                         f" {s['median_latency_ns']:10.1f}")
        lines.append("  Diebold-Mariano on the pinball loss (negative difference = model A better):")
        lines.append(f"    {'A vs B':34} {'pooled diff (bp)':>17} {'DM':>7} {'p':>8}   per stock, significant at 5%: A better / B better / neither")
        for r in (x for x in dm_rows if x["dataset"] == ds and x["scope"] == "pooled"):
            per = [x for x in dm_rows if x["dataset"] == ds and x["scope"] != "pooled" and x["model_a"] == r["model_a"] and x["model_b"] == r["model_b"]]
            a_better = sum(x["p_value"] < 0.05 and x["mean_loss_diff"] < 0 for x in per)
            b_better = sum(x["p_value"] < 0.05 and x["mean_loss_diff"] > 0 for x in per)
            lines.append(f"    {r['model_a'] + ' vs ' + r['model_b']:34} {1e4 * r['mean_loss_diff']:17.4f} {r['dm_stat']:7.2f} {r['p_value']:8.4f}"
                         f"   {a_better} / {b_better} / {len(per) - a_better - b_better}")
        lines.append("")
    lines += [
        "Notes: the pooled test uses, at each target time, the mean loss difference over the stocks (it accounts",
        "for the stocks moving together). p-values are asymptotic (normal), Newey-West variance with Bartlett weights.",
        "Latency: median over the stocks of each run's model-stage mean (batches of 1000 updates); machine-dependent.",
    ]
    (args.out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"tables written to {args.out}")


if __name__ == "__main__":
    main()
