"""Part 3 - data robustness: scenario R1 (a faulty feed) and scenario R2 (genuine shocks).

The two scenarios pull in opposite directions, which is the trade-off this part makes visible:
  R1  the degraded feed (missing bars, duplicated and out-of-order rows, a frozen price, bad ticks)
      must not move the forecasts away from those of the clean feed;
  R2  genuine extreme moves of the clean feed must enter the risk estimate at once, so the VaR
      issued at the shock bar is compared with the same engine without protection.
Each engine is run with and without its protection (EWMA: data-quality layer; GARCH-t: clipping of
standardized shocks at 6 sd). The events come from data/synthetic/scenario.csv, so a regenerated
market needs no code change here.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

from common import (DEGRADED_PRICES, RESULTS, SYNTHETIC, Measurement, Section, median, read_csv, to_float,
                    write_csv, write_text)
from engines import ENGINES, VARIANTS, clean_runs, risk_path
from engines import run as run_engine

DEV_LIMIT = 5.0  # % deviation from the clean-feed forecast counted by R1
FALSE_ALARM = 5.0  # % below the unprotected VaR counted as a hidden move on the clean feed


def scenario_events() -> tuple[list[dict], list[dict]]:
    """(genuine shocks, feed faults) listed in data/synthetic/scenario.csv."""
    _, rows = read_csv(SYNTHETIC / "scenario.csv")
    faults = [r for r in rows if "[degraded feed only]" in r["event"]]
    shocks = [r for r in rows if r not in faults and any(k in r["event"].lower() for k in ("shock", "crash"))]
    return shocks, faults


def _var(out: Path, col: str) -> dict[str, tuple[float, float]]:
    """forecast_time -> (VaR 99%, realized P&L of the target bar)."""
    return {r["forecast_time"]: (to_float(r[col]), to_float(r["realized_pnl"])) for r in read_csv(risk_path(out))[1]}


def deviation(clean: dict, degraded: dict) -> list[tuple[str, float]]:
    return [(t, 100 * abs(degraded[t][0] - clean[t][0]) / clean[t][0]) for t in sorted(set(clean) & set(degraded))
            if clean[t][0] > 0]


def shock_response(series: dict, reference: dict, t_event: str) -> dict:
    """VaR at the first forecast issued at or after the shock, relative to the unprotected engine."""
    times = sorted(t for t in series if t >= t_event and t in reference)
    if not times:
        return {}
    t0 = times[0]
    ratio = series[t0][0] / reference[t0][0]
    delay = next((k for k, t in enumerate(times[:40]) if series[t][0] / reference[t][0] >= 0.99), math.nan)
    affected = sum(abs(series[t][0] / reference[t][0] - 1.0) > 0.005 for t in times[:40])
    pnl = series[t0][1]
    return {"forecast_time": t0, "var99": series[t0][0], "var99_unprotected": reference[t0][0],
            "ratio": ratio, "bars_to_99pct": delay, "bars_affected": affected, "next_bar_pnl": pnl,
            "exception": pnl < -series[t0][0], "exception_unprotected": pnl < -reference[t0][0]}


def _figures(dev: dict, clean: dict, shocks: list, faults: list) -> list[str]:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from plotstyle import INK_2, MUTED, SLIDE, SURFACE, heading, save, variant_style

    figs = []
    # ---- R1: deviation from the clean feed over time, one panel per engine
    engines = [k for k in ENGINES if any(VARIANTS[v].engine == k for v in dev)]
    fig, axes = plt.subplots(len(engines), 1, figsize=SLIDE, sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.75, bottom=0.07, hspace=0.25)
    heading(fig, "R1 - Faulty feed: how far each engine's VaR drifts from the clean-feed VaR",
            "Same synthetic market with missing bars, duplicated and out-of-order rows, a frozen price and three bad "
            "ticks. |VaR 99% (faulty feed) / VaR 99% (clean feed) - 1|,\nmain model of each engine; solid = default "
            "configuration, dashed = protection switched on/off. Log scale; values below 0.01% drawn at 0.01%")
    for ax, ek in zip(axes[:, 0], engines):
        for vkey, (series, share, top) in dev.items():
            v = VARIANTS[vkey]
            if v.engine != ek:
                continue
            xs = [dt.datetime.fromisoformat(t) for t, _ in series]
            ys = [max(d, 0.01) for _, d in series]
            scope = " (worst of 15 stocks per bar)" if ENGINES[ek].per_ticker else ""
            ax.plot(xs, ys, linewidth=1.3, label=f"{v.label}{scope}: {share:.2f}% of forecasts off by more than "
                    f"{DEV_LIMIT:.0f}%, max {top:.0f}%", **variant_style(v))
        ax.set_yscale("log")
        ax.set_ylim(0.01, 20000)
        ax.axhline(DEV_LIMIT, color=MUTED, linewidth=0.8)
        ax.text(1.0, DEV_LIMIT, f"{DEV_LIMIT:.0f}% ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=9, color=INK_2)
        for f in faults:
            x = dt.datetime.fromisoformat(f["timestamp"])
            ax.axvline(x, color=MUTED, linewidth=0.6, alpha=0.6)
        ax.set_ylabel(f"{ENGINES[ek].label}\ndeviation (%)")
        ax.legend(loc="upper left", fontsize=9.5)
    for f in faults:
        x = dt.datetime.fromisoformat(f["timestamp"])
        label = f["event"].replace("[degraded feed only] ", "").replace(" close", "").replace("price frozen for ", "frozen ")
        axes[0][0].annotate(label, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, color=INK_2)
    axes[-1][0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1][0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    figs.append(save(fig, "r1_degraded_feed"))

    # ---- R2: protected / unprotected VaR, bar by bar around each genuine shock
    fig, axes = plt.subplots(1, len(shocks), figsize=SLIDE, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.775, bottom=0.11, wspace=0.08)
    heading(fig, "R2 - Genuine shocks: what each protection costs when the move is real",
            "Clean feed (every move is real). Portfolio VaR 99% of the protected configuration divided by the same "
            "engine without protection, bar by bar\naround the scripted shocks. 1.0 = the protection changes nothing; "
            "below 1.0 = the real move is partly hidden from the risk estimate")
    lo = hi = 1.0
    for j, ev in enumerate(shocks):
        ax = axes[0][j]
        t_ev = ev["timestamp"]
        for ek in engines:
            e = ENGINES[ek]
            prot = next((v for v in clean if VARIANTS[v].engine == ek and v != e.unprotected), None)
            if prot is None:
                continue
            ref = clean[e.unprotected]
            times = sorted(t for t in clean[prot] if t in ref)
            k0 = next(k for k, t in enumerate(times) if t >= t_ev)
            window = times[max(0, k0 - 4): k0 + 9]
            xs = [k - k0 for k in range(max(0, k0 - 4), max(0, k0 - 4) + len(window))]
            ys = [clean[prot][t][0] / ref[t][0] for t in window]
            lo, hi = min(lo, *ys), max(hi, *ys)
            v = VARIANTS[prot]
            ax.plot(xs, ys, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{e.label}: {v.decision}", **variant_style(v))
            k_far = max(range(len(ys)), key=lambda k: abs(ys[k] - 1.0))  # the largest effect
            if abs(ys[k_far] - 1.0) > 0.005:
                ax.annotate(f"{ys[k_far]:.3f}", (xs[k_far], ys[k_far]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=10, color=INK_2)
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel("bars relative to the first forecast issued at or after the shock")
        ax.set_title(ev["event"].split(":")[0].replace("XVAR3 earnings shock", "XVAR3 earnings shock (-9%)")
                     + f"  ({t_ev[5:16]})", fontsize=11.5, fontweight="normal", color=INK_2)
    pad = max(0.02, (hi - lo) * 0.25)
    axes[0][0].set_ylim(lo - pad, hi + pad)
    axes[0][0].set_ylabel("VaR 99% protected / unprotected")
    axes[0][0].legend(loc="lower left", fontsize=9.5)
    figs.append(save(fig, "r2_genuine_shocks"))
    return figs


def shock_in_features(out: Path, ticker: str, t_event: str) -> dict:
    """Single-instrument engines: does a forecast of the same session carry the shock in its realized
    variance (RV >= 0.8 r^2, r the shock bar's return by the shared convention)?"""
    from part_comparability import shared_returns

    ev = dt.datetime.fromisoformat(t_event).replace(tzinfo=dt.timezone.utc)
    r = shared_returns(SYNTHETIC, ticker).get(str(int(ev.timestamp())), math.nan)
    rows = sorted((int(float(x["forecast_time"])), to_float(x["realized_variance"]))
                  for x in read_csv(out / "forecasts.csv")[1] if x["model"] == "Realized-CAViaR")
    same = [(t, rv) for t, rv in rows if t >= ev.timestamp()
            and dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).date() == ev.date()]
    first = dt.datetime.fromtimestamp(same[0][0], tz=dt.timezone.utc).strftime("%H:%M") if same else "none"
    if math.isnan(r):
        return {"contained": True, "bars_to_contain": math.nan, "first_forecast": first, "shock_return": r}
    k = next((i for i, (_, rv) in enumerate(same) if rv >= 0.8 * r * r), None)
    return {"contained": k is not None, "bars_to_contain": k if k is not None else math.nan, "first_forecast": first,
            "shock_return": r}


def _ticker_of(event: str, tickers: list[str], default: str) -> str:
    """The instrument an event names (single-instrument engines replay that one), else the default."""
    return next((t for t in tickers if t in event), default)


def run(fresh: bool = False) -> list[Measurement]:
    shocks, faults = scenario_events()
    tickers = [r["ticker"] for r in read_csv(SYNTHETIC / "assets.csv")[1]]
    variants = [v for v in VARIANTS.values() if ENGINES[v.engine].enabled]
    ms: list[Measurement] = []
    r1 = Section("R1 - faulty feed (prices_15m_degraded.csv) vs clean feed, main-model VaR 99% forecast by forecast")
    r2 = Section("R2 - genuine shocks on the clean feed: VaR issued at the shock bar vs the same engine unprotected")
    rows1, rows2, dev, clean = [], [], {}, {}

    def clean_series(vkey: str, ticker: str | None = None) -> dict:
        e = ENGINES[VARIANTS[vkey].engine]
        return _var(clean_runs(vkey, fresh, reps=1, ticker=ticker)[0], e.var99)

    for v in variants:
        e = ENGINES[v.engine]
        pairs, unmatched, per_time = [], 0, {}
        for tk in (tickers if e.per_ticker else [None]):  # single-instrument engines: every stock of the book
            c = clean_series(v.key, tk)
            d = _var(run_engine(v.key, "synthetic_degraded", SYNTHETIC, prices=DEGRADED_PRICES, fresh=fresh,
                                ticker=tk), e.var99)
            ser = deviation(c, d)
            pairs += [(t, x, tk) for t, x in ser]
            unmatched += len(set(c) ^ set(d))
            for t, x in ser:
                per_time[t] = max(per_time.get(t, 0.0), x)
        clean[v.key] = clean_series(v.key)
        devs = [x for _, x, _ in pairs]
        # a forecast missing from one feed, or not a number, counts as off: it is a failure too
        total = len(devs) + unmatched
        share5 = 100 * (sum(not (x <= DEV_LIMIT) for x in devs) + unmatched) / total
        share1 = 100 * (sum(not (x <= 1.0) for x in devs) + unmatched) / total
        worst = max((p for p in pairs if not math.isnan(p[1])), key=lambda p: p[1])
        where = f"{worst[0]}" + (f" ({worst[2]})" if worst[2] else "")
        unit = f"single-stock forecasts of {len(tickers)} stocks" if e.per_ticker else "portfolio forecasts"
        dev[v.key] = (sorted(per_time.items()), share5, worst[1])
        rows1.append({"variant": v.key, "unit": unit, "forecasts_compared": len(devs), "unmatched_forecasts": unmatched,
                      "share_dev_gt_1pct": round(share1, 3), "share_dev_gt_5pct": round(share5, 3),
                      "median_dev_pct": round(median(devs), 4), "max_dev_pct": round(worst[1], 3), "max_dev_at": where})
        r1.add(f"{v.label:<22} {share5:6.2f}% of {total} {unit} off by >{DEV_LIMIT:.0f}% (>1%: {share1:5.2f}%), "
               f"max {worst[1]:8.2f}% at {where}; {unmatched} issued from one feed only (counted as off)")
        ms += [Measurement("R1", v.key, "degraded_share_dev5_pct", share5, "%",
                           f"{total} {unit} ({unmatched} issued from one feed only, counted as off); >1%: "
                           f"{share1:.2f}%; max {worst[1]:.1f}% at {where}"),
               Measurement("R1", v.key, "r1_max_dev_pct", worst[1], "%"),
               Measurement("R1", v.key, "r1_share_dev1_pct", share1, "%")]
    for v in variants:
        e = ENGINES[v.engine]
        res = []
        for ev in shocks:
            tk = _ticker_of(ev["event"], tickers, e.default_ticker) if e.per_ticker else None
            series, ref = clean_series(v.key, tk), clean_series(e.unprotected, tk)
            r = shock_response(series, ref, ev["timestamp"])
            if r:
                r["instrument"] = tk or "portfolio"
                if e.per_ticker:  # absolute check: do the model's features ever contain the shock?
                    r.update(shock_in_features(clean_runs(v.key, fresh, reps=1, ticker=tk)[0], tk, ev["timestamp"]))
                res.append((ev, r))
        series, ref = clean[v.key], clean[e.unprotected]
        hidden = sum(1 for t in series if t in ref and series[t][0] < ref[t][0] * (1 - FALSE_ALARM / 100))
        for ev, r in res:
            rows2.append({"variant": v.key, "event_time": ev["timestamp"], "event": ev["event"], **r})
        # a shock that never reaches the model's features scores 0 whatever the protection ratio
        worst = min(r["ratio"] if r.get("contained", True) else 0.0 for _, r in res)
        if e.unprotected != v.key:
            protection = f"reference: {VARIANTS[e.unprotected].label}"
        elif e.per_ticker:
            protection = "no protection to switch off; checked instead whether the shock enters the CAViaR features"
        else:
            protection = "no protection to switch off; the filters take every printed bar, so the shock enters at once"

        def describe(ev: dict, r: dict) -> str:
            text = f"{ev['event'].split(':')[0]} ({r['instrument']}): ratio {r['ratio']:.3f}"
            if "contained" in r:
                text += (f", in the features {r['bars_to_contain']} bar(s) later" if r["contained"]
                         else f", NEVER in the features (first forecast of the session {r['first_forecast']})")
            return text + (", VaR exception" if r["exception"] else "")

        r2.add(f"{v.label:<22} " + "; ".join(describe(ev, r) for ev, r in res)
               + f"; clean-feed forecasts more than {FALSE_ALARM:.0f}% below unprotected: {hidden} ({protection})")
        ms += [Measurement("R2", v.key, "shock_min_ratio", worst, "ratio",
                           "; ".join(describe(ev, r) for ev, r in res) + f"; {protection}"),
               Measurement("R2", v.key, "r2_hidden_shocks", sum(not r.get("contained", True) for _, r in res),
                           "shocks", "; ".join(ev["event"].split(":")[0] for ev, r in res
                                                if not r.get("contained", True))),
               Measurement("R2", v.key, "r2_max_effect_pct", max(abs(r["ratio"] - 1) * 100 for _, r in res), "%",
                           "largest |VaR protected / unprotected - 1| at a shock bar"),
               Measurement("R2", v.key, "r2_bars_affected", max(r["bars_affected"] for _, r in res), "bars",
                           "bars after a shock with the protected VaR more than 0.5% away from the unprotected one"),
               Measurement("R2", v.key, "r2_hidden_forecasts", hidden, "forecasts",
                           f"clean-feed forecasts more than {FALSE_ALARM:.0f}% below the unprotected engine")]
    write_csv(RESULTS / "robustness" / "degraded_feed.csv", list(rows1[0].keys()), rows1)
    write_csv(RESULTS / "robustness" / "genuine_shocks.csv", list(rows2[0].keys()), rows2)
    figs = _figures(dev, clean, shocks, faults)
    r1.add("feed faults: " + "; ".join(f"{f['timestamp']} {f['event'].replace('[degraded feed only] ', '')}" for f in faults))
    r2.add("shocks: " + "; ".join(f"{s['timestamp']} {s['event']}" for s in shocks))
    r2.add("figures: " + ", ".join(figs))
    write_text(RESULTS / "robustness" / "report.txt", r1.text() + "\n\n" + r2.text())
    print("   " + "\n   ".join((r1.text() + "\n" + r2.text()).splitlines()))
    return ms
