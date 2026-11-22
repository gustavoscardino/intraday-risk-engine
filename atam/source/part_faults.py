"""Part - fault catalogue (scenario R3): one kind of data problem at a time, on synthetic and real data.

data/tools/inject_faults.py writes one faulty copy of a dataset per kind of problem (missing rows,
halted stocks, duplicated, out-of-order and late rows, frozen quotes, bad ticks, a feed that stops
early, a clock shifted by one hour, corrupt values, an unadjusted split, a stock listed late, and
all the recoverable ones together), with the list of every fault injected. Each engine variant runs
on every copy and is compared, forecast by forecast, with its own run on the clean dataset:

  damaged      share of the forecasts both runs issue whose VaR 99% differs by more than 5%
  lost         share of the clean run's forecasts that the faulty run does not issue
  exceptions   exception rate of the faulty run's VaR against the TRUE outcome (the clean run's
               P&L of the same target bar), next to the clean run's own rate: a protection that
               hides real risk shows up here, a fault that inflates the VaR as a rate near zero
  detected     (EWMA with its data-quality layer) share of the injected faults whose bar and stock
               carry a data-quality flag

Realized-CAViaR runs one stock at a time: every stock of the dataset is run and pooled.
"""
from __future__ import annotations

import concurrent.futures
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from common import DATA, RESULTS, WORK, Measurement, Section, read_csv, table, to_float, write_csv, write_text
from engines import ENGINES, VARIANTS, risk_path
from engines import run as run_engine

DEV_LIMIT = 5.0  # % deviation from the clean-feed VaR counted as damaged
DATASETS = {"synthetic": "XPET3", "b3_real": "PETR4"}  # dataset -> a stock (unused: every stock is run)
FEEDS = WORK / "faults"  # faulty datasets written by inject_faults.py (not versioned)
OUT = RESULTS / "faults"
VARIANT_KEYS = ["ewma", "ewma_nodq", "garch_t", "garch_t_nojump", "garch_t_zclip6", "realized_caviar"]
SCORED = "everything"  # the scenario behind the R3 measure: moderate doses of every recoverable fault
JOBS = 3


def scenarios() -> list[str]:
    out = subprocess.run([sys.executable, str(DATA / "tools" / "inject_faults.py"), "--list"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def build_feeds(names: list[str]) -> None:
    """(Re)writes the faulty datasets; the script is deterministic, so an unchanged run is a no-op for the cache."""
    subprocess.run([sys.executable, str(DATA / "tools" / "inject_faults.py"), "--data", ",".join(DATASETS),
                    "--out", str(FEEDS)], check=True, capture_output=True, text=True)


def tickers(dataset: str) -> list[str]:
    return [r["ticker"] for r in read_csv(DATA / dataset / "assets.csv")[1]]


def run_all(variant: str, dataset: str, folder: Path, tag: str, fresh: bool) -> list[Path | str]:
    """Output folders of one variant on one dataset folder (one per stock for single-stock engines); a
    run that fails is kept as its error message (a crash on a faulty feed is a result)."""
    e = ENGINES[VARIANTS[variant].engine]
    stocks = tickers(dataset) if e.per_ticker else [None]
    outs: list[Path | str] = []
    for t in stocks:
        try:
            outs.append(run_engine(variant, tag, folder, fresh=fresh, ticker=t))
        except Exception as exc:
            lines = [l for l in str(exc).splitlines() if l.strip()]
            outs.append(f"{t + ': ' if t else ''}{(lines[-1] if lines else 'failed')[:140]}")
    return outs


def series(outs: list[Path | str], variant: str) -> dict[tuple[str, int], tuple[float, float]]:
    """(forecast_time, stock index) -> (VaR 99% of the main model, realized P&L of the target bar)."""
    col = ENGINES[VARIANTS[variant].engine].var99
    out = {}
    for k, o in enumerate(outs):
        if isinstance(o, str):
            continue
        for r in read_csv(risk_path(o))[1]:
            out[(r["forecast_time"], k)] = (to_float(r[col]), to_float(r["realized_pnl"]))
    return out


def status(outs: list[Path | str]) -> str:
    failed = [o for o in outs if isinstance(o, str)]
    if not failed:
        return "ok"
    if len(failed) == len(outs):
        return "failed: " + failed[0]
    return f"ok, {len(failed)} of {len(outs)} stocks failed (first: {failed[0]})"


def compare(clean: dict, faulty: dict) -> dict:
    common = [k for k in clean if k in faulty and clean[k][0] > 0 and math.isfinite(faulty[k][0])]
    dev = [100 * abs(faulty[k][0] / clean[k][0] - 1) for k in common]
    scored = [k for k in common if math.isfinite(clean[k][1])]
    exc_clean = sum(clean[k][1] < -clean[k][0] for k in scored)
    exc_faulty = sum(clean[k][1] < -faulty[k][0] for k in scored)  # against the true outcome
    return {
        "compared": len(common),
        "lost_pct": 100 * (1 - len([k for k in clean if k in faulty]) / len(clean)) if clean else math.nan,
        "damaged_pct": 100 * sum(d > DEV_LIMIT for d in dev) / len(dev) if dev else math.nan,
        "median_dev_pct": sorted(dev)[len(dev) // 2] if dev else math.nan,
        "max_dev_pct": max(dev) if dev else math.nan,
        "exc_rate_clean_pct": 100 * exc_clean / len(scored) if scored else math.nan,
        "exc_rate_faulty_pct": 100 * exc_faulty / len(scored) if scored else math.nan,
    }


def detection(out: Path, truth: Path) -> tuple[float, dict]:
    """Share of injected faults whose (bar, stock) carries any data-quality flag; recall per fault kind."""
    flagged = {(r["time"], r["ticker"]) for r in read_csv(out / "dq_events.csv")[1]}
    kinds = defaultdict(lambda: [0, 0])
    for r in read_csv(truth)[1]:
        hit = (r["timestamp"], r["ticker"]) in flagged
        kinds[r["fault"]][0] += 1
        kinds[r["fault"]][1] += hit
    total = sum(v[0] for v in kinds.values())
    return (100 * sum(v[1] for v in kinds.values()) / total if total else math.nan,
            {k: 100 * v[1] / v[0] for k, v in kinds.items()})


def run(fresh: bool = False) -> list[Measurement]:
    names = scenarios()
    build_feeds(names)
    variants = [v for v in VARIANT_KEYS if ENGINES[VARIANTS[v].engine].enabled]
    jobs = [(v, ds, DATA / ds, f"faults_{ds}_clean") for v in variants for ds in DATASETS]
    jobs += [(v, ds, FEEDS / ds / s, f"faults_{ds}_{s}") for v in variants for ds in DATASETS for s in names]
    runs: dict[tuple[str, str, str], list[Path | str]] = {}
    with concurrent.futures.ThreadPoolExecutor(JOBS) as pool:
        for (v, ds, folder, tag), outs in zip(jobs, pool.map(lambda j: run_all(*j, fresh), jobs)):
            runs[(v, ds, tag.split(f"faults_{ds}_", 1)[1])] = outs

    rows, ms = [], []
    clean_rate: dict[tuple[str, str], float] = {}  # exception rate of the whole clean run
    for ds in DATASETS:
        for v in variants:
            clean_outs = runs[(v, ds, "clean")]
            if status(clean_outs) != "ok":
                raise RuntimeError(f"{v} failed on the clean {ds} dataset: {status(clean_outs)}")
            clean = series(clean_outs, v)
            scored = [x for x in clean.values() if x[0] > 0 and math.isfinite(x[1])]
            clean_rate[(ds, v)] = 100 * sum(pnl < -var for var, pnl in scored) / len(scored) if scored else math.nan
            for s in names:
                outs = runs[(v, ds, s)]
                row = {"dataset": ds, "scenario": s, "variant": v, "status": status(outs)}
                if not row["status"].startswith("failed"):
                    row.update(compare(clean, series(outs, v)))
                    if VARIANTS[v].engine == "ewma" and (outs[0] / "dq_events.csv").is_file():
                        row["detected_pct"], per_kind = detection(outs[0], FEEDS / ds / s / "faults.csv")
                        row["detected_by_fault"] = " ".join(f"{k}:{p:.0f}%" for k, p in sorted(per_kind.items()))
                rows.append(row)
                if not row["status"].startswith("failed"):
                    ms.append(Measurement("R3", v, f"damaged {ds}/{s}", row["damaged_pct"], "%",
                                          f"forecasts off by more than {DEV_LIMIT:.0f}% from the clean-feed VaR"))
                else:
                    ms.append(Measurement("R3", v, f"failed {ds}/{s}", 1.0, "", row["status"]))

    header = ["dataset", "scenario", "variant", "status", "compared", "lost_pct", "damaged_pct", "median_dev_pct",
              "max_dev_pct", "exc_rate_clean_pct", "exc_rate_faulty_pct", "detected_pct", "detected_by_fault"]
    write_csv(OUT / "by_scenario.csv", header, rows)

    # R3 (utility tree): damage on the scenario with every recoverable fault together, worst dataset
    for v in variants:
        got = {r["dataset"]: r for r in rows if r["variant"] == v and r["scenario"] == SCORED}
        vals = {ds: r["damaged_pct"] for ds, r in got.items() if not r["status"].startswith("failed")}
        if len(vals) < len(got) or not vals:
            ms.append(Measurement("R3", v, "r3_damaged_pct", 100.0, "%", f"the run failed on the {SCORED} scenario"))
            continue
        worst = max(vals, key=vals.get)
        unit = "single-stock forecasts (every stock)" if ENGINES[VARIANTS[v].engine].per_ticker else "portfolio forecasts"
        ms.append(Measurement("R3", v, "r3_damaged_pct", vals[worst], "%",
                              ", ".join(f"{ds} {x:.1f}%" for ds, x in vals.items())
                              + f" of the {unit} more than {DEV_LIMIT:.0f}% off, "
                              + ", ".join(f"{got[ds]['lost_pct']:.1f}%" for ds in vals) + " not issued"))

    def cell(r, key, fmt="{:.1f}"):
        x = r.get(key)
        if r["status"].startswith("failed"):
            return "crash"
        text = "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)
        return text + ("*" if r["status"] != "ok" else "")  # * = some stocks failed

    sec = Section("Fault catalogue (R3): each engine on a faulty copy of the data, against its own clean-data run")
    sec.add(f"damaged = % of forecasts whose VaR 99% moves more than {DEV_LIMIT:.0f}% from the clean-data VaR;")
    sec.add("exceptions = exception rate of the VaR against the TRUE outcome (clean P&L), clean run -> faulty run;")
    sec.add("lost = % of the clean run's forecasts not issued. Realized-CAViaR: every stock, pooled;")
    sec.add("* = some stocks of a single-stock engine failed (see by_scenario.csv); crash = the run failed.")
    for ds in DATASETS:
        sec.add("")
        sec.add(f"{ds}: damaged % (lost %)")
        trows = []
        for s in names:
            by_v = {r["variant"]: r for r in rows if r["dataset"] == ds and r["scenario"] == s}
            trows.append([s] + [f"{cell(by_v[v], 'damaged_pct')} ({cell(by_v[v], 'lost_pct')})" for v in variants])
        sec.lines += table(["scenario", *[VARIANTS[v].label for v in variants]], trows)
        sec.add("")
        sec.add(f"{ds}: exception rate % at 99% against the true outcome, faulty run (clean run in the first line)")
        trows = [["clean data"] + [f"{clean_rate[(ds, v)]:.2f}" for v in variants]]
        for s in names:
            by_v = {r["variant"]: r for r in rows if r["dataset"] == ds and r["scenario"] == s}
            trows.append([s] + [cell(by_v[v], "exc_rate_faulty_pct", "{:.2f}") for v in variants])
        sec.lines += table(["scenario", *[VARIANTS[v].label for v in variants]], trows)
        if "ewma" in variants:
            sec.add("")
            sec.add(f"{ds}: faults flagged by the EWMA data-quality layer (bar and stock carry a flag)")
            trows = [[s, cell(r, "detected_pct"), r.get("detected_by_fault", "")]
                     for s in names for r in rows if r["dataset"] == ds and r["scenario"] == s and r["variant"] == "ewma"]
            sec.lines += table(["scenario", "detected %", "by fault"], trows, "lrl")
    write_text(OUT / "report.txt", sec.text())
    print(sec.text())
    return ms
