"""Part 4 - performance: scenarios P1 (per-bar latency), P2 (scalability) and P3 (recalibration cost).

P1 reads the per-bar engine time (update_us) that both portfolio replays record around the same work:
fills, data quality, filter updates and the VaR of the next bar. Outside that timer: the EWMA
confirmation of the previous bar's suspect prices (step 1 of its loop, before t0) and the GARCH-t
Monte Carlo revaluation of the option book (reported apart as mc_us). End-of-session bars are excluded because
they also run the daily refit, which P3 measures from daily_summary.csv. Each figure is the median
over the three clean replays shared with Rp1.

P2 runs each engine's own benchmark program three times and parses its table. The two programs do
not report the same statistics (EWMA: p50/p95/p99 per bar; GARCH-t: median per bar only), so the
comparison uses the median, the one statistic both report. Parsers live in BENCH_PARSERS: if a
benchmark changes its output, only its parser needs an update.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from common import RESULTS, SYNTHETIC, Measurement, Section, median, percentile, read_csv, to_float, write_csv, write_text
from engines import ENGINES, REPS, VARIANTS, clean_runs, default_variants, run_benchmark


def parse_ewma_bench(text: str) -> dict[int, dict]:
    """'assets  bar p50 (us)  bar p95 (us)  bar p99 (us) refit p50 (ms) refit max (ms)' table."""
    out = {}
    for m in re.finditer(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", text, re.M):
        n, p50, p95, p99, rp50, rmax = m.groups()
        out[int(n)] = {"bar_p50_us": float(p50), "bar_p95_us": float(p95), "bar_p99_us": float(p99),
                       "refit_ms": float(rp50)}
    return out


def parse_garch_bench(text: str) -> dict[int, dict]:
    """'portfolio engine' table: assets, refit 1 thr, refit all thr, bar update+VaR (median), bars/s."""
    out = {}
    part = text.split("portfolio engine", 1)[-1]
    for m in re.finditer(r"^\s*(\d+)\s+([\d.]+) ms\s+([\d.]+) ms\s+([\d.]+) us\s+(\d+)\s*$", part, re.M):
        n, r1, rall, p50, _ = m.groups()
        out[int(n)] = {"bar_p50_us": float(p50), "refit_ms": float(rall), "refit_1thread_ms": float(r1)}
    return out


BENCH_PARSERS = {"ewma": parse_ewma_bench, "garch_t": parse_garch_bench}


def _spread(variant: str, per_run: list[dict]) -> list[Measurement]:
    """Lowest and highest value across the repeated runs: the scorecard calls a threshold that falls inside
    this range (with a margin) borderline instead of met or not met."""
    out = []
    for k in ("p50", "p99"):
        xs = [p[k] for p in per_run]
        out += [Measurement("P1", variant, f"latency_{k}_us_min", min(xs), "us", f"{len(xs)} runs"),
                Measurement("P1", variant, f"latency_{k}_us_max", max(xs), "us", f"{len(xs)} runs")]
    return out


def _eod_rows(rows: list[dict]) -> set[str]:
    """Forecast times issued at the last bar of each session (they include the daily refit)."""
    last: dict[str, str] = {}
    for r in rows:
        d = r["forecast_time"][:10]
        last[d] = max(last.get(d, ""), r["forecast_time"])
    return set(last.values())


def _figure(lat: dict, bench: dict) -> str:
    import matplotlib.pyplot as plt

    from plotstyle import ENGINE_COLOR, INK_2, MUTED, SLIDE, SURFACE, heading, save

    fig, (a1, a2) = plt.subplots(1, 2, figsize=SLIDE, gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.11, wspace=0.22)
    heading(fig, "P1/P2 - Portfolio engines: VaR updated in microseconds; GARCH-t has the heavier tail",
            "Left: per-bar engine latency in the replay of the synthetic market (15 stocks, intraday bars, end-of-"
            "session refit excluded), median of three runs.\nRight: median per-bar latency from each engine's "
            "benchmark program as the book grows (log-log), median of three runs")
    keys = list(lat)
    stats = ["p50", "p95", "p99"]
    width = 0.36
    for k, vkey in enumerate(keys):
        v = VARIANTS[vkey]
        xs = [i + (k - (len(keys) - 1) / 2) * width for i in range(len(stats))]
        ys = [lat[vkey][s] for s in stats]
        a1.bar(xs, ys, width=width - 0.04, color=ENGINE_COLOR[v.engine], label=v.label)
        for x, y in zip(xs, ys):
            a1.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=10, color=INK_2)
    a1.axhline(10, color=MUTED, linewidth=0.8)
    a1.text(-0.45, 10, "P1 limit 10 us", ha="left", va="bottom", fontsize=9.5, color=INK_2)
    a1.set_xticks(range(len(stats)), stats)
    a1.set_ylabel("per-bar latency (us)")
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    for ek, by_n in bench.items():
        ns = sorted(by_n)
        ys = [by_n[n]["bar_p50_us"] for n in ns]
        a2.plot(ns, ys, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                color=ENGINE_COLOR[ek], label=ENGINES[ek].label)
        a2.annotate(f"{ys[-1]:.0f} us", (ns[-1], ys[-1]), xytext=(-10, 8 if ek == "garch_t" else -14),
                    textcoords="offset points", ha="right", fontsize=10, color=INK_2)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.axhline(1000, color=MUTED, linewidth=0.8)
    a2.text(200, 1000, "P2 limit 1 ms (200 assets) ", ha="right", va="bottom", fontsize=9, color=INK_2)
    a2.set_xticks([15, 50, 200], ["15", "50", "200"])
    a2.set_xlabel("assets in the book")
    a2.set_ylabel("median per-bar latency (us)")
    a2.legend(loc="lower right")
    a2.grid(True, axis="both")
    return save(fig, "p1_p2_latency")


MODEL_STAGE_KEYS = ("mean_ns", "p50_ns", "p95_ns", "p99_ns", "state_bytes", "updates_per_second")


def _technique(model: str) -> str:
    return "ewma" if model.startswith("EWMA") else "garch_t" if model.startswith("GARCH") else "realized_caviar"


def model_stage(fresh: bool) -> tuple[list[Measurement], Section, list[dict], dict]:
    """Single-instrument engines time each model at the model stage (benchmark.csv of the replay)."""
    ms: list[Measurement] = []
    s = Section("P1/P3/P4 - single-instrument engines: model-stage cost of each model under one measurement boundary "
                "(update + quantile + monetary VaR, timed in batches of 1000; median of three runs)")
    rows, agg_all = [], {}
    for v in default_variants():
        e = ENGINES[v.engine]
        if not e.per_ticker:
            continue
        runs = clean_runs(v.key, fresh)
        per_model: dict[str, list[dict]] = defaultdict(list)
        refits = []
        for out in runs:
            for r in read_csv(out / "benchmark.csv")[1]:
                per_model[r["model"]].append({k: to_float(r[k]) for k in MODEL_STAGE_KEYS})
            by_t: dict[str, float] = defaultdict(float)
            for r in read_csv(out / "params_daily.csv")[1]:
                by_t[r["effective_time"]] += to_float(r["fit_ms"])
            refits.append((median(list(by_t.values())), max(by_t.values()), len(by_t)))
        agg = {m: {k: median([x[k] for x in xs]) for k in MODEL_STAGE_KEYS} for m, xs in per_model.items()}
        agg_all[v.key] = agg
        for m, a in agg.items():
            rows.append({"variant": v.key, "model": m, **{k: round(x, 3) for k, x in a.items()}})
            s.add(f"{m:<16} mean {a['mean_ns']:6.1f} ns  p50 {a['p50_ns']:6.1f} ns  p95 {a['p95_ns']:6.1f} ns  "
                  f"p99 {a['p99_ns']:6.1f} ns  {a['updates_per_second'] / 1e6:5.1f} M updates/s  state "
                  f"{a['state_bytes']:.0f} bytes")
        main_name = "Realized-CAViaR" if "Realized-CAViaR" in agg else next(iter(agg))
        main = agg[main_name]
        refit_p50, refit_max = median([r[0] for r in refits]), median([r[1] for r in refits])
        s.add(f"daily calibration of the three models (sum of fit times): p50 {refit_p50:.1f} ms, max {refit_max:.1f} ms "
              f"({refits[0][2]} calibrations, {e.default_ticker})")
        s.add("percentiles are of batch means (1000 updates per batch), which average away the per-update tail: they "
              "compare the models with each other, not with the per-bar p99 of the portfolio engines")
        boundary = ("model stage of one instrument (no features, CSV or scoring); percentiles of 1000-update batch "
                    "means, so not comparable with a per-bar p99")
        # P1 asks for a per-bar p99: this engine does not measure one, so P1 is not applicable to it
        ms += [Measurement("P1", v.key, "latency_p99_us", math.nan, "n/a",
                           f"{boundary}: CAViaR p50 {main['p50_ns']:.1f} ns, p99 of batch means {main['p99_ns']:.1f} ns"),
               Measurement("P1", v.key, "model_stage_p50_us", main["p50_ns"] / 1e3, "us", boundary),
               *[Measurement("P1", v.key, f"model_stage_p50_ns:{m}", a["p50_ns"], "ns", "median of batch means")
                 for m, a in agg.items()],
               *[Measurement("P4", v.key, f"state_bytes:{m}", a["state_bytes"], "bytes") for m, a in agg.items()],
               Measurement("P3", v.key, "refit_max_ms", refit_max, "ms",
                           f"daily calibration of CAViaR + EWMA-N + GARCH-t on {e.default_ticker}: p50 "
                           f"{refit_p50:.1f} ms, max {refit_max:.1f} ms"),
               Measurement("P3", v.key, "refit_p50_ms", refit_p50, "ms"),
               Measurement("P4", v.key, "state_bytes_per_asset", main["state_bytes"], "bytes",
                           "persistent model state per instrument: " + ", ".join(
                               f"{m} {a['state_bytes']:.0f} B" for m, a in agg.items()))]
    return ms, s, rows, agg_all


def _figure_model_stage(agg: dict) -> str:
    import matplotlib.pyplot as plt

    from plotstyle import ENGINE_COLOR, INK_2, SLIDE, heading, save

    fig, (a1, a2) = plt.subplots(1, 2, figsize=SLIDE, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.11, wspace=0.25)
    heading(fig, "Model-stage cost of the three models under one measurement boundary",
            "realized_caviar's matched benchmark: same instrument, same forecasts, update + quantile + monetary VaR "
            "timed in batches of 1000 (median of three runs).\nThis is the boundary to use for the cost side of the "
            "research question; the portfolio engines time a whole bar cycle of 15 stocks instead")
    models = list(agg)
    stats = ["p50_ns", "p95_ns", "p99_ns"]
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        xs = [i + (k - (len(models) - 1) / 2) * width for i in range(len(stats))]
        ys = [agg[m][st] for st in stats]
        a1.bar(xs, ys, width=width - 0.04, color=ENGINE_COLOR[_technique(m)], label=m)
        for x, y in zip(xs, ys):
            a1.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=10, color=INK_2)
    a1.set_xticks(range(len(stats)), [st.replace("_ns", "") for st in stats])
    a1.set_ylabel("ns per forecast")
    a1.legend(loc="upper left")
    ys = [agg[m]["state_bytes"] for m in models]
    a2.barh(range(len(models)), ys, height=0.5, color=[ENGINE_COLOR[_technique(m)] for m in models])
    for i, y in enumerate(ys):
        a2.text(y, i, f" {y:.0f} B", va="center", fontsize=10, color=INK_2)
    a2.set_yticks(range(len(models)), models)
    a2.invert_yaxis()
    a2.set_xlabel("persistent model state per instrument (bytes)")
    a2.grid(True, axis="x")
    a2.grid(False, axis="y")
    a2.set_xlim(0, max(ys) * 1.3)
    return save(fig, "p1_p4_model_stage")


def run(fresh: bool = False) -> list[Measurement]:
    ms: list[Measurement] = []
    s = Section("P1 - per-bar latency (replay, 15 assets, intraday bars; from the scored forecasts of risk_15m.csv), "
                "P3 - daily refit (replay)")
    rows, lat = [], {}
    for v in default_variants():
        if ENGINES[v.engine].per_ticker:  # timed at the model stage instead: see model_stage()
            continue
        runs = clean_runs(v.key, fresh)
        per_run = []
        for out in runs:
            _, fr = read_csv(out / "risk_15m.csv")
            eod = _eod_rows(fr)
            us = [to_float(r["update_us"]) for r in fr if r["forecast_time"] not in eod]
            _, daily = read_csv(out / "daily_summary.csv")
            refit = [to_float(r["refit_ms"]) for r in daily]
            per_run.append({"n": len(us), "p50": percentile(us, 50), "p95": percentile(us, 95),
                            "p99": percentile(us, 99), "max": max(us), "refit_p50": median(refit),
                            "refit_max": max(refit), "refits": len(refit)})
        agg = {k: median([p[k] for p in per_run]) for k in ("p50", "p95", "p99", "max", "refit_p50", "refit_max")}
        lat[v.key] = agg
        for i, p in enumerate(per_run, 1):
            rows.append({"variant": v.key, "run": i, "bars": p["n"], "p50_us": round(p["p50"], 3),
                         "p95_us": round(p["p95"], 3), "p99_us": round(p["p99"], 3), "max_us": round(p["max"], 2),
                         "refits": p["refits"], "refit_p50_ms": round(p["refit_p50"], 3),
                         "refit_max_ms": round(p["refit_max"], 3)})
        spread = [p["p99"] for p in per_run]
        s.add(f"{v.label:<14} p50 {agg['p50']:6.2f} us  p95 {agg['p95']:6.2f} us  p99 {agg['p99']:7.2f} us  "
              f"max {agg['max']:8.1f} us   refit p50 {agg['refit_p50']:7.2f} ms, max {agg['refit_max']:7.2f} ms   "
              f"(p99 across runs {min(spread):.2f}-{max(spread):.2f} us)")
        # slowest bars: is the tail made of bars at which a fill was applied? (median over runs, like the max)
        no_fill_max, mc = [], []
        for k, out in enumerate(runs):
            _, fr = read_csv(out / "risk_15m.csv")
            eod = _eod_rows(fr)
            times = sorted(r["forecast_time"] for r in fr)
            fill_bars = {next((t for t in times if t >= f["timestamp"]), "")
                         for f in read_csv(SYNTHETIC / "fills.csv")[1]}
            intraday = [(to_float(r["update_us"]), r["forecast_time"]) for r in fr if r["forecast_time"] not in eod]
            no_fill_max.append(max(u for u, t in intraday if t not in fill_bars))
            mc += [to_float(r["mc_us"]) for r in fr if "mc_us" in r and r["mc_us"]]
            if k == 0:
                slow = sorted(intraday, reverse=True)[:5]
                s.add(f"{'':<14} slowest bars (run 1): " + ", ".join(
                    f"{u:.0f} us{' (fill)' if t in fill_bars else ''}" for u, t in slow))
        s.add(f"{'':<14} without fill bars: max {median(no_fill_max):.1f} us (median of runs)")
        ms.append(Measurement("P1", v.key, "latency_max_without_fill_bars_us", median(no_fill_max), "us",
                              f"{sum(t in fill_bars for _, t in slow)} of the 5 slowest bars of run 1 applied a fill"))
        if mc:  # GARCH-t revalues the option book by Monte Carlo at every bar, outside update_us
            ms.append(Measurement("P1", v.key, "mc_revaluation_p50_us", median(mc), "us",
                                  "per-bar Monte Carlo revaluation of the option book, timed apart from update_us"))
            s.add(f"{'':<14} option book revalued by Monte Carlo at every bar, outside update_us: median "
                  f"{median(mc) / 1e3:.2f} ms per bar")
        ms += [Measurement("P1", v.key, "latency_p99_us", agg["p99"], "us",
                           f"p50 {agg['p50']:.2f}, p95 {agg['p95']:.2f}, p99 {agg['p99']:.2f}, max {agg['max']:.1f} us; "
                           f"median of {len(runs)} runs ({per_run[0]['n']} intraday bars each)"),
               Measurement("P1", v.key, "latency_p50_us", agg["p50"], "us"),
               Measurement("P1", v.key, "latency_p95_us", agg["p95"], "us"),
               Measurement("P3", v.key, "refit_max_ms", agg["refit_max"], "ms",
                           f"daily refit in the replay: p50 {agg['refit_p50']:.2f} ms, max {agg['refit_max']:.2f} ms "
                           f"({per_run[0]['refits']} refits, median of {len(runs)} runs)"),
               Measurement("P3", v.key, "refit_p50_ms", agg["refit_p50"], "ms"),
               Measurement("P1", v.key, "latency_max_us", agg["max"], "us"),
               # P4 asks for a constant state per instrument: a portfolio engine keeps the estimation
               # window of every stock and an n x n correlation, so its state is not one
               Measurement("P4", v.key, "state_bytes_per_asset", math.nan, "n/a",
                           "portfolio engine: the state per stock grows with the estimation window (40 sessions of "
                           "returns kept for the daily refit) and with the book (a row of the n x n correlation); "
                           "the per-instrument filter states are compared in the matched model-stage benchmark")]
        ms += _spread(v.key, per_run)
    # the non-default variants: the latency cost of switching a decision on or off (same three runs)
    for v in VARIANTS.values():
        if v.default or not ENGINES[v.engine].enabled or ENGINES[v.engine].per_ticker:
            continue
        per_run = []
        for i, out in enumerate(clean_runs(v.key, fresh), 1):
            _, fr = read_csv(out / "risk_15m.csv")
            eod = _eod_rows(fr)
            us = [to_float(r["update_us"]) for r in fr if r["forecast_time"] not in eod]
            per_run.append({"p50": percentile(us, 50), "p99": percentile(us, 99)})
            rows.append({"variant": v.key, "run": i, "bars": len(us), "p50_us": round(percentile(us, 50), 3),
                         "p95_us": round(percentile(us, 95), 3), "p99_us": round(percentile(us, 99), 3),
                         "max_us": round(max(us), 2)})
        p50, p99 = median([p["p50"] for p in per_run]), median([p["p99"] for p in per_run])
        s.add(f"{v.label:<14} p50 {p50:6.2f} us  p99 {p99:7.2f} us  (median of {len(per_run)} runs; {v.decision})")
        ms += [Measurement("P1", v.key, "latency_p50_us", p50, "us", f"median of {len(per_run)} runs"),
               Measurement("P1", v.key, "latency_p99_us", p99, "us", f"median of {len(per_run)} runs"),
               *_spread(v.key, per_run)]
    write_csv(RESULTS / "performance" / "latency_replay.csv",
              list(dict.fromkeys(k for r in rows for k in r)), rows)

    # ---- P2: benchmark programs
    s2 = Section("P2 - scalability (each engine's benchmark program, median of three runs)")
    bench_rows, bench = [], {}
    for v in default_variants():
        e = ENGINES[v.engine]
        parser = BENCH_PARSERS.get(e.key)
        if e.per_ticker:
            ms.append(Measurement("P2", v.key, "scale200_p50_us", math.nan, "n/a",
                                  "single-instrument engine: there is no multi-asset mode; a book of N stocks is N "
                                  "independent runs and no portfolio VaR"))
            continue
        if parser is None:
            continue
        tables = []
        for i in range(1, REPS + 1):
            text = run_benchmark(e.key, f"rep{i}", fresh)
            (RESULTS / "performance").mkdir(parents=True, exist_ok=True)
            if i == 1:
                (RESULTS / "performance" / f"benchmark_{e.key}.txt").write_text(text, encoding="utf-8")
            t = parser(text)
            if not t:
                raise RuntimeError(f"could not parse the {e.key} benchmark output: update BENCH_PARSERS['{e.key}'] "
                                   "in atam/source/part_performance.py")
            tables.append(t)
        by_n = defaultdict(dict)
        for n in tables[0]:
            for k in tables[0][n]:
                by_n[n][k] = median([t[n][k] for t in tables if n in t])
        bench[e.key] = dict(by_n)
        for n in sorted(by_n):
            bench_rows.append({"engine": e.key, "assets": n, **{k: round(x, 3) for k, x in by_n[n].items()}})
            s2.add(f"{e.label:<8} {n:>4} assets: median per bar {by_n[n]['bar_p50_us']:8.2f} us"
                   + (f", p99 {by_n[n]['bar_p99_us']:8.2f} us" if "bar_p99_us" in by_n[n] else ", p99 not reported")
                   + f"; refit {by_n[n]['refit_ms']:8.2f} ms")
        top = max(by_n)
        ms.append(Measurement("P2", v.key, "scale200_p50_us", by_n[top]["bar_p50_us"], "us",
                              f"{top} assets, median per bar; "
                              + ("p99 " + f"{by_n[top]['bar_p99_us']:.1f} us" if "bar_p99_us" in by_n[top]
                                 else "the benchmark reports no p99")))
    write_csv(RESULTS / "performance" / "scaling_benchmark.csv",
              sorted({k for r in bench_rows for k in r}, key=lambda k: (k not in ("engine", "assets"), k)), bench_rows)
    fig = _figure(lat, bench)
    s2.add(f"figure: {fig}")
    ms3, s3, rows3, agg3 = model_stage(fresh)
    ms += ms3
    if rows3:
        write_csv(RESULTS / "performance" / "model_stage_benchmark.csv", list(rows3[0].keys()), rows3)
        s3.add(f"figure: {_figure_model_stage(next(iter(agg3.values())))}")
    text = s.text() + "\n\n" + s2.text() + ("\n\n" + s3.text() if rows3 else "")
    write_text(RESULTS / "performance" / "report.txt", text)
    print("   " + "\n   ".join(text.splitlines()))
    return ms
