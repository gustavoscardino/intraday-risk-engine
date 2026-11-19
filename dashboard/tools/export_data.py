"""Exports the repository's results to the dashboard's JSON files (dashboard/data/).

Reads what the engines already wrote and versioned:
  realized_caviar/comparison/   matched comparison over every stock (tools/compare_all.py)
  ewma/results/, garch_t/results/   backtests of the portfolio engines
  atam/results/                 architecture evaluation (scorecard, findings, robustness)
  data/*/assets.csv             the stocks of each dataset
and copies the architecture diagrams (architecture/*.svg) to dashboard/assets/.
Rerun it whenever those results change:

    python dashboard/tools/export_data.py
"""
import csv
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "dashboard" / "data"
ASSETS = REPO / "dashboard" / "assets"


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not next(iter(r.values()), "").startswith("#")]


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return None if x != x else (int(x) if x.is_integer() and "." not in str(v) and "e" not in str(v).lower() else x)


def typed(rs):
    return [{k: num(v) for k, v in r.items()} for r in rs]


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  data/{name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("exporting dashboard data")

    # ---- datasets
    datasets = {}
    for ds in ("synthetic", "b3_real"):
        assets = rows(REPO / "data" / ds / "assets.csv")
        datasets[ds] = [{"ticker": a["ticker"], "name": a.get("name", ""), "sector": a.get("sector", "")} for a in assets]
    write("datasets.json", datasets)

    # ---- matched comparison (native engine, every stock)
    comp = REPO / "realized_caviar" / "comparison"
    report = (comp / "report.txt").read_text(encoding="utf-8").splitlines()
    code_line = next((l for l in report if l.startswith("Code:")), "")
    write("comparison.json", {
        "summary": typed(rows(comp / "summary.csv")),
        "by_stock": typed(rows(comp / "by_stock.csv")),
        "by_hour": typed(rows(comp / "by_hour.csv")),
        "diebold_mariano": typed(rows(comp / "diebold_mariano.csv")),
        "windows": {"synthetic": 60, "b3_real": 30},
        "code": code_line.replace("Code: ", ""),
    })

    # ---- portfolio engines (their own backtests)
    portfolio = {}
    for engine, runs in (("ewma", ("synthetic", "synthetic_degraded", "synthetic_degraded_nodq", "b3_real")),
                         ("garch_t", ("synthetic", "synthetic_degraded", "b3_real"))):
        for run in runs:
            path = REPO / engine / "results" / run / "backtest_summary.csv"
            if path.exists():
                portfolio.setdefault(engine, {})[run] = typed(rows(path))
    write("portfolio.json", portfolio)

    # ---- ATAM
    atam = REPO / "atam" / "results"
    write("atam.json", {
        "utility_tree": typed(rows(REPO / "atam" / "utility_tree.csv")),
        "scorecard": typed(rows(atam / "scorecard.csv")),
        "findings": typed(rows(atam / "findings.csv")),
        "degraded_feed": typed(rows(atam / "robustness" / "degraded_feed.csv")),
        "model_stage": typed(rows(atam / "performance" / "model_stage_benchmark.csv")),
    })

    # ---- diagrams
    for name in ("00_visao_geral", "01_fluxo_de_dados", "03_ciclo_15min", "04_modulos_garch_t", "05_realized_caviar", "02_modulos"):
        src = REPO / "architecture" / f"{name}.svg"
        if src.exists():
            shutil.copyfile(src, ASSETS / f"{name}.svg")
            print(f"  assets/{name}.svg")

    write("meta.json", {"generated": date.today().isoformat(), "commit": commit})


if __name__ == "__main__":
    main()
