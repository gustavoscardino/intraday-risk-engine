"""Part 1 - correctness: scenario C1 (no look-ahead) and scenario Rp1 (reproducibility).

C1 replays alternative histories for three cut times T and checks that nothing issued at or before
T changes:
  perturbation   the prices after T follow a different random path (same file length);
  truncation     the file ends at T, as in a live feed in which the future does not exist yet;
  truncation, fixed window   the same, with the estimation window given on the command line.
Compared: every forecast column of the engine's forecast files issued at or before T (outcomes such
as the realized P&L only when the target bar is also at or before T), and the calibration files of
the refits run at or before T; both lists are declared per engine in engines.py. Timing columns are
never compared. End-of-run aggregates (backtest summary, last correlation) are outside C1.

Rp1 runs the clean replay three times and compares every output CSV cell by cell.
"""
from __future__ import annotations

import datetime as dt
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from common import RESULTS, SYNTHETIC, WORK, Measurement, Section, read_csv, rel, write_csv, write_text
from engines import ENGINES, Engine, clean_runs, default_variants, epoch_to_iso, risk_path, window_days
from engines import run as run_engine

ENGINE_INPUTS = ("assets.csv", "portfolio_initial.csv", "fills.csv", "options.csv")
# realized after the forecast: compared only when the target bar is at or before the cut
OUTCOME = {"realized_pnl", "book_realized_pnl", "target_return", "evaluated", "target_time"}
PERTURB_SIGMA = 0.005  # log-price step per bar of the alternative future
CUTS = ((0.25, "12:00:00", "mid-session"), (0.50, "17:00:00", "end of session (daily refit)"),
        (0.75, "10:15:00", "first bar of a session"))
TESTS = ("perturbation", "truncation", "truncation, fixed window")


def is_timing(col: str) -> bool:
    return col.endswith(("_us", "_ms", "_ns")) or col in ("seconds", "updates_per_second")


# ----------------------------------------------------------------------------- alternative histories


def _slug(t: str) -> str:
    return t.replace("-", "").replace(":", "").replace(" ", "_")


def _fresh_dir(name: str) -> Path:
    d = WORK / "datasets" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def perturbed_dataset(T: str) -> Path:
    """data/synthetic with every price after T moved along a seeded random path, per ticker."""
    d = _fresh_dir(f"perturb_{_slug(T)}")
    for name in ENGINE_INPUTS:
        if (SYNTHETIC / name).is_file():
            shutil.copy2(SYNTHETIC / name, d / name)
    header, rows = read_csv(SYNTHETIC / "prices_15m.csv")
    rng = random.Random(f"atam-c1-{T}")  # str seeds are hashed deterministically (sha512)
    level: dict[str, float] = {}
    for r in rows:
        if r["timestamp"] <= T:
            continue
        lf = level.get(r["ticker"], 0.0) + rng.gauss(0.0, PERTURB_SIGMA)
        level[r["ticker"]] = lf
        f = math.exp(lf)
        for c in ("open", "high", "low", "close"):  # one factor per row keeps low <= open, close <= high
            r[c] = f"{float(r[c]) * f:.2f}"
    write_csv(d / "prices_15m.csv", header, rows)
    return d


def truncated_dataset(T: str) -> Path:
    """data/synthetic as a live feed sees it at T: no price and no fill after T."""
    d = _fresh_dir(f"truncate_{_slug(T)}")
    for name in ENGINE_INPUTS:
        if name != "fills.csv" and (SYNTHETIC / name).is_file():
            shutil.copy2(SYNTHETIC / name, d / name)
    for name in ("prices_15m.csv", "fills.csv"):
        header, rows = read_csv(SYNTHETIC / name)
        write_csv(d / name, header, [r for r in rows if r["timestamp"] <= T])
    return d


def cut_times(base: Path) -> list[tuple[str, str]]:
    _, rows = read_csv(risk_path(base))
    dates = sorted({r["forecast_time"][:10] for r in rows})
    _, prices = read_csv(SYNTHETIC / "prices_15m.csv")
    stamps = {r["timestamp"] for r in prices}
    cuts = []
    for frac, hhmm, label in CUTS:
        t = f"{dates[round(frac * (len(dates) - 1))]} {hhmm}"
        if t not in stamps:
            raise ValueError(f"cut {t} is not a bar of the synthetic feed")
        cuts.append((t, label))
    return cuts


# ----------------------------------------------------------------------------- comparisons


@dataclass
class Diff:
    compared: int = 0
    changed: int = 0
    missing_rows: int = 0  # issued at or before T in the reference run, absent from the test run
    extra_rows: int = 0  # issued at or before T in the test run, absent from the reference run
    later: int = 0  # forecasts issued after T (perturbation power)
    later_changed: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def violations(self) -> int:
        return self.changed + self.missing_rows + self.extra_rows

    def note(self, s: str) -> None:
        if len(self.examples) < 5:
            self.examples.append(s)


def _eod_bars() -> dict[str, str]:
    """Last bar of every session of the clean feed (the bar at which the daily refit runs)."""
    _, prices = read_csv(SYNTHETIC / "prices_15m.csv")
    eod: dict[str, str] = {}
    for r in prices:
        eod[r["timestamp"][:10]] = max(eod.get(r["timestamp"][:10], ""), r["timestamp"])
    return eod


def compare_up_to(e: Engine, base: Path, test: Path, T: str, eod: dict[str, str]) -> Diff:
    d = Diff()
    main_var = e.var99
    for name in e.forecast_files:
        if not (base / name).is_file():
            continue
        hb, rb = read_csv(base / name)
        _, rt = read_csv(test / name) if (test / name).is_file() else ([], [])
        other = {r["forecast_time"]: r for r in rt}
        cols = [c for c in hb if c != "forecast_time" and not is_timing(c)]
        for r in rb:
            t = other.get(r["forecast_time"])
            if r["forecast_time"] > T:
                if name == e.risk_file and t is not None:
                    d.later += 1
                    d.later_changed += r[main_var] != t[main_var]
                continue
            target_after = r["target_time"] > T
            if t is None:
                if not target_after:  # truncation may only drop forecasts whose target lies after T
                    d.missing_rows += 1
                    d.note(f"{name} {r['forecast_time']}: forecast missing")
                continue
            for c in cols:
                if c in OUTCOME and target_after:
                    continue
                d.compared += 1
                if r[c] != t.get(c):
                    d.changed += 1
                    d.note(f"{name} {r['forecast_time']} {c}: {r[c]} -> {t.get(c)}")
        known = {r["forecast_time"] for r in rb}
        for r in rt:
            if r["forecast_time"] <= T and r["forecast_time"] not in known:
                d.extra_rows += 1
                d.note(f"{name} {r['forecast_time']} -> {r['target_time']}: forecast not in the reference run")
    for name, key, tcol, kind in e.calibration_files:
        if not (base / name).is_file():
            continue
        hb, rb = read_csv(base / name)
        _, rt = read_csv(test / name) if (test / name).is_file() else ([], [])
        other = {tuple(r[k] for k in key): r for r in rt}
        cols = [c for c in hb if c not in key and not is_timing(c)]
        for r in rb:
            when = eod.get(r[tcol], "9999") if kind == "date" else epoch_to_iso(r[tcol]) or "9999"
            if when > T:  # calibrations run after the cut may legitimately differ
                continue
            t = other.get(tuple(r[k] for k in key))
            if t is None:
                d.missing_rows += 1
                d.note(f"{name} {' '.join(r[k] for k in key)}: refit missing")
                continue
            for c in cols:
                d.compared += 1
                if r[c] != t.get(c):
                    d.changed += 1
                    d.note(f"{name} {' '.join(r[k] for k in key)} {c}: {r[c]} -> {t.get(c)}")
    return d


def compare_runs(a: Path, b: Path) -> tuple[int, int, int, list[str]]:
    """Every output CSV of two runs, cell by cell (timing columns excluded)."""
    files = sorted(p.name for p in a.glob("*.csv"))
    compared = differing = 0
    notes: list[str] = []
    if files != sorted(p.name for p in b.glob("*.csv")):
        notes.append("different sets of output files")
    for name in files:
        ha, ra = read_csv(a / name)
        hb, rb = read_csv(b / name) if (b / name).is_file() else ([], [])
        if ha != hb or len(ra) != len(rb):
            differing += max(len(ra), len(rb)) * max(len(ha), 1)
            notes.append(f"{name}: header or row count differs")
            continue
        cols = [c for c in ha if not is_timing(c)]
        for x, y in zip(ra, rb):
            for c in cols:
                compared += 1
                if x[c] != y[c]:
                    differing += 1
                    if len(notes) < 5:
                        notes.append(f"{name} {c}: {x[c]} vs {y[c]}")
    return len(files), compared, differing, notes


# ----------------------------------------------------------------------------- figure


def _figure(curves: dict, cuts: list[tuple[str, str]]) -> str:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from engines import VARIANTS
    from plotstyle import GRID, INK_2, MUTED, SLIDE, SURFACE, heading, save, variant_style

    fig, axes = plt.subplots(len(curves), len(cuts), figsize=SLIDE, sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.07, wspace=0.06, hspace=0.32)
    heading(fig, "C1 - No look-ahead: forecasts issued before the cut never change",
            "Prices after the cut T replaced by a different random path. Relative change of each forecast's "
            "VaR 99% (main model) versus the original replay")
    top = max(max(ys) for by_cut in curves.values() for xs, ys, *_ in by_cut.values())
    for i, (vkey, by_cut) in enumerate(curves.items()):
        v = VARIANTS[vkey]
        for j, (T, label) in enumerate(cuts):
            ax = axes[i][j]
            xs, ys, n_before, ch_before, n_after, ch_after = by_cut[T]
            tcut = dt.datetime.fromisoformat(T)
            ax.axvspan(xs[0], tcut, color=GRID, alpha=0.35, linewidth=0)
            ax.plot(xs, ys, linewidth=1.2, **variant_style(v))
            ax.axvline(tcut, color=MUTED, linewidth=0.8)
            ax.text(0.02, 0.96, f"before T: {ch_before} of {n_before} changed\nafter T: {ch_after} of {n_after} changed",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=INK_2, zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.92})
            if i == 0:
                ax.set_title(f"{label}  (T = {T[5:16]})", fontsize=11.5, fontweight="normal", color=INK_2)
            if j == 0:
                ax.set_ylabel(f"{v.label}\n|change| of VaR 99% (%)")
            ax.set_ylim(0, top * 1.35)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    return save(fig, "c1_no_lookahead")


# ----------------------------------------------------------------------------- part


def run(fresh: bool = False) -> list[Measurement]:
    ms: list[Measurement] = []
    eod = _eod_bars()
    rows_c1, rows_rp = [], []
    curves: dict[str, dict] = {}
    s = Section("C1 - no look-ahead (forecasts at or before a cut time T must not depend on what follows T)")
    rp = Section("Rp1 - reproducibility (the same replay, three runs, every output cell compared)")
    base_runs = {v.key: clean_runs(v.key, fresh) for v in default_variants()}
    cuts = cut_times(next(iter(base_runs.values()))[0])
    for v in default_variants():
        e = ENGINES[v.engine]
        base = base_runs[v.key][0]
        win = window_days(base)
        totals = {t: Diff() for t in TESTS}
        curves[v.key] = {}
        for T, label in cuts:
            tests = {
                "perturbation": run_engine(v.key, f"c1_perturb_{_slug(T)}", perturbed_dataset(T), fresh=fresh),
                "truncation": run_engine(v.key, f"c1_truncate_{_slug(T)}", truncated_dataset(T), fresh=fresh),
                "truncation, fixed window": run_engine(v.key, f"c1_truncate_win_{_slug(T)}", truncated_dataset(T),
                                                fresh=fresh, extra=("--window-days", str(win))),
            }
            for test, out in tests.items():
                d = compare_up_to(e, base, out, T, eod)
                tot = totals[test]
                tot.compared += d.compared
                tot.changed += d.changed
                tot.missing_rows += d.missing_rows
                tot.extra_rows += d.extra_rows
                tot.later += d.later
                tot.later_changed += d.later_changed
                tot.examples += d.examples[: max(0, 3 - len(tot.examples))]
                rows_c1.append({"variant": v.key, "cut": T, "cut_position": label, "test": test,
                                "window_days": window_days(out), "compared_cells": d.compared,
                                "changed_cells": d.changed, "missing_rows": d.missing_rows,
                                "extra_rows": d.extra_rows, "examples": " | ".join(d.examples[:3]),
                                "later_forecasts": d.later if test == "perturbation" else "",
                                "later_changed_pct": round(100 * d.later_changed / d.later, 2)
                                if test == "perturbation" and d.later else ""})
                if test == "perturbation":  # relative change of the main VaR per forecast, for the figure
                    b = {r["forecast_time"]: r for r in read_csv(risk_path(base))[1]}
                    o = {r["forecast_time"]: r for r in read_csv(risk_path(out))[1]}
                    xs, ys = [], []
                    n_before = n_after = ch_before = ch_after = 0
                    for t in sorted(set(b) & set(o)):
                        x, y = float(b[t][e.var99]), float(o[t][e.var99])
                        xs.append(dt.datetime.fromisoformat(t))
                        ys.append(100 * abs(y - x) / x if x else 0.0)
                        if t <= T:
                            n_before += 1
                            ch_before += b[t][e.var99] != o[t][e.var99]
                        else:
                            n_after += 1
                            ch_after += b[t][e.var99] != o[t][e.var99]
                    curves[v.key][T] = (xs, ys, n_before, ch_before, n_after, ch_after)
        p, tr, tw = totals["perturbation"], totals["truncation"], totals["truncation, fixed window"]
        power = 100 * p.later_changed / p.later if p.later else math.nan
        ms += [
            Measurement("C1", v.key, "lookahead_changed_cells", p.violations + tr.violations, "cells",
                        f"perturbation {p.violations}; truncation {tr.violations} (default window resolved from "
                        f"the file length); truncation with --window-days {win}: {tw.changed} changed cells, "
                        f"{tw.missing_rows} missing and {tw.extra_rows} extra rows"),
            Measurement("C1", v.key, "c1_perturbation_changed", p.violations, "cells"),
            Measurement("C1", v.key, "c1_truncation_changed", tr.violations, "cells"),
            Measurement("C1", v.key, "c1_truncation_fixed_window_changed", tw.violations, "cells"),
            Measurement("C1", v.key, "c1_compared_cells", p.compared + tr.compared + tw.compared, "cells"),
            Measurement("C1", v.key, "c1_power_pct", power, "%",
                        "share of forecasts issued after T whose VaR 99% changed under perturbation"),
        ]
        scope = f" ({e.default_ticker})" if e.per_ticker else ""
        s.add(f"{v.label}{scope}: {p.compared + tr.compared + tw.compared} cells compared over {len(cuts)} cuts")
        for test, d in totals.items():
            s.add(f"  {test:<25} changed {d.changed:>6} cells, missing {d.missing_rows:>3} rows, "
                  f"extra {d.extra_rows:>3} rows"
                  + (f"   (test power: {power:.1f}% of later forecasts changed)" if test == "perturbation" else ""))
            for ex in d.examples[:3]:
                s.add(f"      e.g. {ex}")

        # ---- Rp1
        runs = base_runs[v.key]
        diff_total = comp_total = 0
        for other in runs[1:]:
            nf, comp, diff, notes = compare_runs(runs[0], other)
            diff_total += diff
            comp_total += comp
            rows_rp.append({"variant": v.key, "run_a": rel(runs[0]), "run_b": rel(other), "files": nf,
                            "compared_cells": comp, "differing_cells": diff, "notes": "; ".join(notes)})
        ms += [Measurement("Rp1", v.key, "repro_differing_cells", diff_total, "cells",
                           f"{len(runs)} runs, {comp_total} cells compared, timing columns excluded"),
               Measurement("Rp1", v.key, "repro_compared_cells", comp_total, "cells")]
        rp.add(f"{v.label}: {len(runs)} runs, {comp_total} cells compared, {diff_total} differ")

    s.add("")
    s.add("cuts: " + "; ".join(f"{T} ({label})" for T, label in cuts))
    s.add(f"perturbation: log-price random walk, sd {PERTURB_SIGMA:.3f} per bar, seeded by the cut time")
    write_csv(RESULTS / "correctness" / "no_lookahead.csv", list(rows_c1[0].keys()), rows_c1)
    write_csv(RESULTS / "correctness" / "reproducibility.csv", list(rows_rp[0].keys()), rows_rp)
    fig = _figure(curves, cuts)
    s.add(f"figure: {fig}")
    write_text(RESULTS / "correctness" / "report.txt", s.text() + "\n\n" + rp.text())
    print("   " + "\n   ".join((s.text() + "\n" + rp.text()).splitlines()))
    return ms
