"""Replay the same single-stock feed faults with all four VaR models."""

import argparse
import csv
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("EWMA-N", "EWMA-t", "GARCH-t", "Realized-CAViaR")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def replay(binary, work, dataset, scenario, prices, ticker):
    output = work / "replay" / dataset / scenario / ticker
    command = [
        str(binary), "--data", str(ROOT / "data" / dataset), "--ticker", ticker,
        "--prices", str(prices), "--out", str(output), "--window-days", "40",
        "--refit", "daily", "--diurnal", "on", "--confidence", ".99",
        "--exposure", "100000",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{dataset}/{scenario}/{ticker}: {result.stderr.strip()}")
    forecasts = {model: {} for model in MODELS}
    for row in read_csv(output / "risk_15m.csv"):
        key = (ticker, row["forecast_time"], row["target_time"])
        model = row["model"]
        var = float(row["var"])
        if key in forecasts[model] or not math.isfinite(var):
            raise ValueError(f"{dataset}/{scenario}/{ticker}: invalid or duplicate VaR for {model} at {key}")
        forecasts[model][key] = (var, float(row["realized_pnl"]))
    if not all(forecasts.values()):
        raise ValueError(f"{dataset}/{scenario}/{ticker}: replay did not emit forecasts for all models")
    return dataset, scenario, forecasts


def summarize(dataset, scenario, model, clean, faulty):
    paired = clean.keys() & faulty.keys()
    comparable = {key for key in paired if clean[key][0] > 0}
    shifted = sum(abs(faulty[key][0] / clean[key][0] - 1) > .05 for key in comparable)
    clean_only = len(clean.keys() - faulty.keys())
    faulty_only = len(faulty.keys() - clean.keys())
    scored = {key for key in comparable if math.isfinite(clean[key][1])}
    clean_exceptions = sum(clean[key][1] < -clean[key][0] for key in scored)
    faulty_exceptions = sum(clean[key][1] < -faulty[key][0] for key in scored)
    return {
        "dataset": dataset, "scenario": scenario, "model": model,
        "clean": len(clean), "faulty": len(faulty), "paired": len(paired),
        "comparable": len(comparable), "shifted": shifted,
        "clean_only": clean_only, "faulty_only": faulty_only, "scored": len(scored),
        "clean_exceptions": clean_exceptions, "faulty_exceptions": faulty_exceptions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="Release replay executable")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True, help="destination CSV")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="matched-feed-faults-") as temporary:
        work = Path(temporary)
        subprocess.run([
            sys.executable, str(ROOT / "data/tools/inject_faults.py"), "--only",
            "everything", "--out", str(work / "feeds"),
        ], check=True)
        scenarios = {
            dataset: {
                "clean": ROOT / "data" / dataset / "prices_15m.csv",
                "combined": work / "feeds" / dataset / "everything" / "prices_15m.csv",
            }
            for dataset in ("synthetic", "b3_real")
        }
        scenarios["synthetic"]["faulty_feed"] = ROOT / "data/synthetic/prices_15m_degraded.csv"
        series = {
            (dataset, scenario): {model: {} for model in MODELS}
            for dataset, feeds in scenarios.items() for scenario in feeds
        }
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(replay, args.binary.resolve(), work, dataset, scenario, prices, row["ticker"])
                for dataset, feeds in scenarios.items()
                for scenario, prices in feeds.items()
                for row in read_csv(ROOT / "data" / dataset / "assets.csv")
            ]
            for future in futures:
                dataset, scenario, forecasts = future.result()
                for model in MODELS:
                    series[dataset, scenario][model].update(forecasts[model])
    rows = [
        summarize(dataset, scenario, model, series[dataset, "clean"][model], series[dataset, scenario][model])
        for dataset, feeds in scenarios.items()
        for scenario in feeds if scenario != "clean"
        for model in MODELS
    ]
    with args.out.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
