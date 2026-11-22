"""Part 2 - comparability: scenario Cp1 (every model receives the same inputs).

The EWMA and GARCH-t engines both compute the same baseline, an EWMA with lambda = 0.94, from their
own ingestion and return pipelines. If the pipelines are equivalent, that baseline (and the
exposure and realized P&L of every forecast) must coincide forecast by forecast. Any difference
points to an input-handling decision that makes the model comparison unfair.

Pairs compared, on the synthetic market and on the real B3 data:
  default configurations      EWMA (data-quality layer and the shared jump filter) vs GARCH-t (the
                              shared jump filter);
  same data policy            EWMA (layer off) vs GARCH-t (filter off): every print enters both.
The baseline is one implementation in both engines (common/, RiskMetricsModel), so a difference comes
from the inputs: a default-pair difference that the same-policy pair does not show is attributed to
the data protections; the differences left with the same policy come from the pipelines' handling of
missing prints (the bar of the last partial print before the forecast is cited).
"""
from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path

from common import B3_REAL, RESULTS, SYNTHETIC, Measurement, Section, read_csv, to_float, write_csv, write_text
from engines import ENGINES, VARIANTS, clean_runs, default_variants
from engines import run as run_engine

TOL = 0.006  # values are printed with at least two decimals: half a cent plus rounding
FIELDS = ("baseline_var99", "baseline_sigma", "net_exposure", "realized_pnl")
PAIRS = (("ewma", "garch_t", "default configurations"), ("ewma_nodq", "garch_t_nojump", "same data policy"))
PROTECTION = "data protection"  # cause of a difference due to the EWMA data-quality layer or the GARCH-t jump filter


def _col(variant: str, field: str) -> str:
    e = ENGINES[VARIANTS[variant].engine]
    return {"baseline_var99": e.baseline_var99, "baseline_sigma": e.baseline_sigma}.get(field, field)


def partial_bars(data_dir: Path) -> dict[str, int]:
    """Bars in which some assets did not print: timestamp -> number of assets missing."""
    n = len(read_csv(data_dir / "assets.csv")[1])
    counts: dict[str, int] = {}
    for r in read_csv(data_dir / "prices_15m.csv")[1]:
        counts[r["timestamp"]] = counts.get(r["timestamp"], 0) + 1
    return {t: n - c for t, c in counts.items() if c < n}


def compare_pair(a_key: str, a_out: Path, b_key: str, b_out: Path, partial: dict[str, int]) -> dict:
    ra = {r["forecast_time"]: r for r in read_csv(a_out / "risk_15m.csv")[1]}
    rb = {r["forecast_time"]: r for r in read_csv(b_out / "risk_15m.csv")[1]}
    common = sorted(set(ra) & set(rb))
    dq_bars = set()
    for out in (a_out, b_out):
        if (out / "dq_events.csv").is_file():
            dq_bars |= {r["time"] for r in read_csv(out / "dq_events.csv")[1]}
        if (out / "jump_events.csv").is_file():
            dq_bars |= {r["time"] for r in read_csv(out / "jump_events.csv")[1]}
    partial_times = sorted(partial)
    mismatch, series = [], []
    for t in common:
        diffs = {f: abs(to_float(ra[t][_col(a_key, f)]) - to_float(rb[t][_col(b_key, f)])) for f in FIELDS}
        va, vb = to_float(ra[t][_col(a_key, "baseline_var99")]), to_float(rb[t][_col(b_key, "baseline_var99")])
        series.append((dt.datetime.fromisoformat(t), 100 * (va - vb) / vb if vb else 0.0))
        if any(d > TOL for d in diffs.values()):
            before = [p for p in partial_times if p <= t]
            cause = ("data-quality event" if t in dq_bars else
                     f"missing-data policy (partial bar {before[-1]}: {partial[before[-1]]} assets missing)" if before
                     else "unexplained")
            mismatch.append({"forecast_time": t, "fields": " ".join(f for f, d in diffs.items() if d > TOL),
                             "baseline_var99_a": va, "baseline_var99_b": vb,
                             "rel_diff_pct": round(100 * (va - vb) / vb, 4) if vb else "",
                             "explained_by_dq_event": t in dq_bars, "cause": cause})
    return {"common": len(common), "only_a": len(set(ra) - set(rb)), "only_b": len(set(rb) - set(ra)),
            "mismatch": mismatch, "series": series}


def _causes(res: dict) -> str:
    counts: dict[str, int] = {}
    for m in res["mismatch"]:
        key = m["cause"].split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"


MATCHED_FIELDS = ("target_time", "target_return", "realized_pnl", "exposure", "evaluated")


def shared_returns(data_dir: Path, ticker: str) -> dict[str, float]:
    """Bar returns of one stock by the repository's shared convention (as in garch_t market_data and ewma):
    first bar of a session ln(close / open), then close-to-close between consecutive bars of the session.
    Keyed by the bar's close time in epoch seconds of the exchange wall clock (realized_caviar's clock)."""
    import datetime as _dt

    rows = [r for r in read_csv(data_dir / "prices_15m.csv")[1] if r["ticker"] == ticker]
    rows.sort(key=lambda r: r["timestamp"])
    out, prev = {}, None
    for r in rows:
        t = _dt.datetime.fromisoformat(r["timestamp"]).replace(tzinfo=_dt.timezone.utc)
        c, o = to_float(r["close"]), to_float(r["open"])
        if prev is None or prev[0].date() != t.date():
            ret = math.log(c / o) if c > 0 and o > 0 else math.nan
        elif (t - prev[0]).total_seconds() == 900 and prev[1] > 0 and c > 0:
            ret = math.log(c / prev[1])
        else:
            ret = math.nan  # a gap inside the session: no single-bar return
        out[str(int(t.timestamp()))] = ret
        prev = (t, c)
    return out


def matched_check(out: Path, data_dir: Path, ticker: str) -> dict:
    """realized_caviar: (1) its target returns against the returns re-derived from the prices with the shared
    convention, (2) whether its models share every forecast time, target and outcome (by construction)."""
    _, rows = read_csv(out / "forecasts.csv")
    ref = shared_returns(data_dir, ticker)
    main = [r for r in rows if r["model"] == "Realized-CAViaR" and r["evaluated"] == "1"]
    bad_returns = [r for r in main if not (abs(to_float(r["target_return"]) - ref.get(r["target_time"], math.nan))
                                           <= 1e-12)]
    by_time: dict[str, list[dict]] = {}
    for r in rows:
        by_time.setdefault(r["forecast_time"], []).append(r)
    models = sorted({r["model"] for r in rows})
    bad = [t for t, rs in by_time.items()
           if len(rs) != len(models) or any(len({r[f] for r in rs}) > 1 for f in MATCHED_FIELDS)]
    text = (out / "report.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"targets start between (\d+):(\d+) and (\d+):(\d+)", text)
    first = int(m.group(1)) + int(m.group(2)) / 60 if m else math.nan
    return {"models": models, "times": len(by_time), "mismatch": len(bad), "first_target_hour": first,
            "range": f"{m.group(1)}:{m.group(2)}-{m.group(3)}:{m.group(4)}" if m else "unknown",
            "targets": len(main), "return_mismatch": len(bad_returns)}


def _figure(results: dict, events: dict, partials: dict) -> str:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from plotstyle import ENGINE_COLOR, INK_2, MUTED, SLIDE, SURFACE, heading, save

    datasets = list(results)
    fig, axes = plt.subplots(len(datasets), 1, figsize=SLIDE, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.775, bottom=0.07, hspace=0.42)
    differ = sum(len(results[ds]["ewma"]["mismatch"]) + len(results[ds]["ewma_nodq"]["mismatch"]) for ds in results)
    heading(fig, "Cp1 - Comparability: " + ("the two pipelines give the shared baseline the same inputs" if differ == 0
                                             else "where the two pipelines still give the baseline different inputs"),
            "Shared EWMA(0.94) baseline VaR 99% (one implementation, common/), EWMA engine vs GARCH-t engine, forecast "
            "by forecast. Blue: default\nconfigurations (the shared jump filter in both; EWMA also its data-quality "
            "layer); gray: both protections off")
    for ax, ds in zip(axes[:, 0], datasets):
        aligned, default = results[ds]["ewma_nodq"], results[ds]["ewma"]
        xs = [x for x, _ in aligned["series"]]
        ys = [y for _, y in aligned["series"]]
        ax.plot(xs, ys, color=MUTED, linewidth=1.4,
                label=f"same data policy: {len(aligned['mismatch'])} of {aligned['common']} differ")
        pts = [(x, y) for x, y in default["series"] if abs(y) > 1e-9]
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=40, color=ENGINE_COLOR["ewma"], zorder=4,
                   edgecolors=SURFACE, linewidths=1.5,
                   label=f"default configurations: {len(default['mismatch'])} of {default['common']} differ")
        span = (xs[-1] - xs[0]).total_seconds()
        busy = [dt.datetime.fromisoformat(t) for t in partials.get(ds, {})] if aligned["mismatch"] else []
        tagged = [(x, y, events.get(ds, {}).get(x.strftime("%Y-%m-%d %H:%M:%S"))) for x, y in pts]
        tagged = [t for t in tagged if t[2] and all(abs((t[0] - b).days) > 14 for b in busy)]
        shown: list = []  # label the largest differences only, one per week, so that labels do not overlap
        for x, y, tag in sorted(tagged, key=lambda t: -abs(t[1])):
            if len(shown) == 4 or any(abs((x - z).days) < 7 for z in shown):
                continue
            shown.append(x)
            right = (x - xs[0]).total_seconds() > 0.93 * span
            ax.annotate(f"{tag} (DQ)", (x, y), xytext=(-5 if right else 5, -4 if y < 0 else 4),
                        textcoords="offset points", fontsize=9, color=INK_2, ha="right" if right else "left",
                        va="top" if y < 0 else "bottom")
        if aligned["mismatch"]:
            i = min(range(len(ys)), key=lambda k: ys[k])
            t0, k = sorted(partials.get(ds, {}).items(), key=lambda kv: -kv[1])[0]
            ax.annotate(f"{t0[5:16]}: {k} of 15 stocks did not print at the open\n"
                        "the engines' missing-data policies differ", (xs[i], ys[i]), xytext=(-60, 30),
                        textcoords="offset points", ha="right", va="center", fontsize=9, color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_title({"synthetic": "Synthetic market (150 sessions, 15 stocks)",
                      "b3_real": "Real B3 data (87 sessions, 15 stocks)"}.get(ds, ds),
                     fontsize=11.5, fontweight="normal", color=INK_2)
        ax.set_ylabel("difference of\nbaseline VaR 99% (%)")
        lim = max([abs(y) for _, y in default["series"]] + [1.0]) * 1.4
        ax.set_ylim(-lim, lim)
        ax.legend(loc="upper left", ncols=2)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return save(fig, "cp1_comparability")


def run(fresh: bool = False) -> list[Measurement]:
    sources = {"synthetic": SYNTHETIC, "b3_real": B3_REAL}
    datasets = {
        "synthetic": {k: clean_runs(k, fresh, reps=1)[0] for k in ("ewma", "ewma_nodq", "garch_t", "garch_t_nojump")},
        "b3_real": {k: run_engine(k, "b3_real", B3_REAL, fresh=fresh)
                    for k in ("ewma", "ewma_nodq", "garch_t", "garch_t_nojump")},
    }
    s = Section("Cp1 - comparability (shared EWMA(0.94) baseline, exposure and realized P&L, forecast by forecast)")
    rows, results, events, partials = [], {}, {}, {}
    for ds, outs in datasets.items():
        results[ds] = {}
        ev = {}
        if (outs["ewma"] / "dq_events.csv").is_file():
            for r in read_csv(outs["ewma"] / "dq_events.csv")[1]:
                if r["flag"] == "SUSPECT":
                    ev[r["time"]] = r["ticker"]
        events[ds] = ev
        partial = partial_bars(sources[ds])
        partials[ds] = partial
        s.add(f"{ds}: {len(partial)} bars in which some assets did not print"
              + (": " + ", ".join(f"{t} ({k} missing)" for t, k in sorted(partial.items())[:4]) if partial else ""))
        for a, b, _ in PAIRS:
            results[ds][a] = compare_pair(a, outs[a], b, outs[b], partial)
        # differential attribution: the two pairs differ only by the data protections, so a
        # default-pair difference that the same-policy pair also shows comes from the pipelines'
        # missing-data policies, and any other one from the protections
        same_policy = {m["forecast_time"]: m["cause"] for m in results[ds]["ewma_nodq"]["mismatch"]}
        for m in results[ds]["ewma"]["mismatch"]:
            m["cause"] = same_policy.get(m["forecast_time"], PROTECTION)
        for a, b, label in PAIRS:
            res = results[ds][a]
            s.add(f"  {VARIANTS[a].label} vs {VARIANTS[b].label} ({label}): {len(res['mismatch'])} of {res['common']} "
                  f"forecasts differ; forecasts only in one engine: {res['only_a']} / {res['only_b']}")
            s.add(f"      causes: {_causes(res)}")
            for m in res["mismatch"]:
                rows.append({"dataset": ds, "pair": f"{a} vs {b}", **m})
            worst = sorted(res["mismatch"], key=lambda m: -abs(m["rel_diff_pct"] or 0))[:3]
            for m in worst:
                s.add(f"      e.g. {m['forecast_time']}  baseline VaR 99% {m['rel_diff_pct']:+.2f}%  ({m['cause']}"
                      + (f": {ev[m['forecast_time']]}" if m["forecast_time"] in ev else "") + ")")
    write_csv(RESULTS / "comparability" / "baseline_mismatches.csv",
              ["dataset", "pair", "forecast_time", "fields", "baseline_var99_a", "baseline_var99_b", "rel_diff_pct",
               "cause"], rows)
    fig = _figure(results, events, partials)
    s.add(f"tolerance {TOL} (printed precision); figure: {fig}")

    syn, b3 = results["synthetic"], results["b3_real"]
    detail = (f"synthetic: {len(syn['ewma']['mismatch'])} of {syn['ewma']['common']} differ ({_causes(syn['ewma'])}); "
              f"same data policy: {len(syn['ewma_nodq']['mismatch'])}. B3 real data: "
              f"{len(b3['ewma']['mismatch'])} of {b3['ewma']['common']} default, {len(b3['ewma_nodq']['mismatch'])} "
              f"same policy ({_causes(b3['ewma_nodq'])})")
    ms = []
    for v in default_variants():
        e = ENGINES[v.engine]
        if e.per_ticker:  # single-instrument engine: its returns against the shared convention, its models inside
            checks = {"synthetic": matched_check(clean_runs(v.key, fresh, reps=1)[0], SYNTHETIC, e.default_ticker),
                      "b3_real": matched_check(run_engine(v.key, "b3_real", B3_REAL, fresh=fresh, ticker="PETR4"),
                                               B3_REAL, "PETR4")}
            c0 = checks["synthetic"]
            s.add(f"{v.label} (one instrument per run; synthetic {e.default_ticker}, B3 PETR4): models "
                  f"{', '.join(c0['models'])}")
            for ds, c in checks.items():
                s.add(f"  {ds}: {c['return_mismatch']} of {c['targets']} evaluated target returns differ from the "
                      f"returns re-derived from prices_15m.csv (shared convention); {c['mismatch']} of {c['times']} "
                      f"forecast times where the models do not share one target record (by construction); "
                      f"evaluated targets {c['range']}")
            bad = sum(c["return_mismatch"] + c["mismatch"] for c in checks.values())
            ms += [Measurement("Cp1", v.key, "baseline_mismatches", bad, "forecasts",
                               "; ".join(f"{ds}: {c['targets'] - c['return_mismatch']} of {c['targets']} target "
                                         f"returns equal the shared convention" for ds, c in checks.items())
                               + "; the three models share one target record per forecast by construction"),
                   Measurement("Cp1", v.key, "evaluated_from_hour", c0["first_target_hour"], "h",
                               f"evaluated targets {c0['range']}: the opening hour is outside the comparison")]
            continue
        def count(res: dict) -> int:  # forecasts issued by one engine only count as differences too
            return len(res["mismatch"]) + res["only_a"] + res["only_b"]

        ms.append(Measurement("Cp1", v.key, "baseline_mismatches", count(syn["ewma"]), "forecasts", detail))
        ms.append(Measurement("Cp1", v.key, "baseline_mismatches_same_policy", count(syn["ewma_nodq"]), "forecasts"))
        ms.append(Measurement("Cp1", v.key, "baseline_mismatches_b3_default", count(b3["ewma"]), "forecasts"))
        ms.append(Measurement("Cp1", v.key, "baseline_mismatches_b3_same_policy", count(b3["ewma_nodq"]), "forecasts"))
        ms.append(Measurement("Cp1", v.key, "b3_same_policy_max_diff_pct",
                              max([abs(m["rel_diff_pct"] or 0) for m in b3["ewma_nodq"]["mismatch"]] + [0.0]), "%"))
        ms.append(Measurement("Cp1", v.key, "b3_partial_bar_max_missing", max(partials["b3_real"].values(), default=0),
                              "assets", f"of {len(read_csv(B3_REAL / 'assets.csv')[1])} assets"))
    write_text(RESULTS / "comparability" / "report.txt", s.text())
    print("   " + "\n   ".join(s.text().splitlines()))
    return ms
