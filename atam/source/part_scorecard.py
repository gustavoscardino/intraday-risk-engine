"""Part 6 - scorecard: joins the measurements with the utility tree and writes the ATAM outputs.

  results/scorecard.csv          every scenario x engine: measured value, threshold, status
  results/findings.csv           risks, non-risks, sensitivity points and trade-off points
  results/report.txt             the consolidated report (this summary + every part's report)
  results/latex/*.tex            the utility tree, the scorecard and the findings, ready to \\input
  results/figures/atam_*.png     utility tree, scorecard and trade-off map, for slides

Statuses: MET / NOT MET against the threshold of the utility tree; REPORTED for a measure without a
threshold (op 'report'); N/A when the architecture has no such capability (a measurement with unit
'n/a'); PENDING when nothing was measured.

Findings are rules over the measurements (FINDING_RULES): each one is written only while the
evidence supports it, with the numbers of the current run. When an engine changes and a problem
disappears, its finding disappears from the next report without editing this file.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from common import RESULTS, UTILITY_TREE, Section, read_all_measurements, read_csv, write_csv, write_text
from engines import ENGINES, VARIANTS

ENGINE_ORDER = ["ewma", "garch_t", "realized_caviar"]
PARTS = ["correctness", "comparability", "robustness", "faults", "performance", "modifiability"]


# ----------------------------------------------------------------------------- data access


class M:
    """Measurement lookup: m[variant, metric] -> value (nan when absent)."""

    def __init__(self):
        self.by = {(x.variant, x.metric): x for x in read_all_measurements()}

    def __getitem__(self, key) -> float:
        x = self.by.get(key)
        return x.value if x else math.nan

    def unit(self, variant: str, metric: str) -> str:
        x = self.by.get((variant, metric))
        return x.unit if x else ""

    def detail(self, variant: str, metric: str) -> str:
        x = self.by.get((variant, metric))
        return x.detail if x else ""

    def has(self, *keys) -> bool:
        return all(k in self.by for k in keys)


def default_variant(engine: str) -> str | None:
    return next((v.key for v in VARIANTS.values() if v.engine == engine and v.default), None)


def defaults() -> list[str]:
    return [default_variant(e) for e in ENGINE_ORDER if ENGINES[e].enabled and default_variant(e)]


def portfolio_defaults() -> list[str]:
    return [v for v in defaults() if not ENGINES[VARIANTS[v].engine].per_ticker]


def single_defaults() -> list[str]:
    return [v for v in defaults() if ENGINES[VARIANTS[v].engine].per_ticker]


def lab(variant: str) -> str:
    return ENGINES[VARIANTS[variant].engine].label


def names(vs: list[str]) -> str:
    labels = [lab(v) for v in vs]
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


NOISE_MARGIN = 1.25  # run-to-run variation of timings observed on the reference laptop (turbo boost)


def status(value: float, op: str, threshold: str, unit: str = "", lo: float = math.nan, hi: float = math.nan) -> str:
    """MET / NOT MET against the threshold; PARTIAL when the threshold falls inside the range of the
    repeated runs widened by NOISE_MARGIN, i.e. when another run could flip the verdict."""
    if unit == "n/a":
        return "N/A"
    if math.isnan(value):
        return "PENDING"
    if op == "report":
        return "REPORTED"
    if op == "pending":
        return "PENDING"
    t = float(threshold)
    if not (math.isnan(lo) or math.isnan(hi)) and lo / NOISE_MARGIN <= t <= hi * NOISE_MARGIN:
        return "PARTIAL"
    ok = {"<=": value <= t, ">=": value >= t, "==": value == t, "<": value < t, ">": value > t}[op]
    return "MET" if ok else "NOT MET"


def latency_cost(m: M, on: str, off: str) -> tuple[float, bool]:
    """Median p50 difference between two variants, and whether it exceeds the run-to-run spread."""
    diff = m[on, "latency_p50_us"] - m[off, "latency_p50_us"]
    noise = max(m[k, "latency_p50_us_max"] - m[k, "latency_p50_us_min"] for k in (on, off))
    return diff, abs(diff) > noise


def fmt(v: float, unit: str) -> str:
    if unit == "n/a":
        return "n/a"
    if math.isnan(v):
        return "-"
    if unit == "yes/no":
        return "yes" if v >= 1 else "no"
    if unit == "ratio":
        return f"{v:.3f}"
    if unit == "%":
        return f"{v:.2f}%"
    if abs(v) >= 100 or float(v).is_integer():
        return f"{v:,.0f} {unit}".replace(",", " ")
    if abs(v) < 0.1:
        return f"{v:.3f} {unit}"
    return f"{v:.2f} {unit}"


# ----------------------------------------------------------------------------- findings


@dataclass
class Finding:
    kind: str  # risk | non-risk | sensitivity | trade-off
    id: str
    title: str
    text: str
    scenarios: list[str] = field(default_factory=list)


def _f_no_value_lookahead(m: M):
    vs = [v for v in defaults() if m.has((v, "c1_perturbation_changed"))]
    if vs and all(m[v, "c1_perturbation_changed"] == 0 for v in vs):
        power = min(m[v, "c1_power_pct"] for v in vs)
        return Finding("non-risk", "N1", "No look-ahead in the data path",
                       f"Replacing every price after a cut time leaves all earlier forecasts and calibrations unchanged "
                       f"in {names(vs)} ({sum(m[v, 'c1_compared_cells'] for v in vs):,.0f} cells over three cuts); the "
                       f"same test changes {power:.0f}% of the later forecasts (lowest engine)"
                       + (", so it would have caught a leak." if power >= 99 else ", so its power is limited."),
                       ["C1"])


def _f_deterministic(m: M):
    vs = [v for v in defaults() if m.has((v, "repro_differing_cells"))]
    if vs and all(m[v, "repro_differing_cells"] == 0 for v in vs):
        return Finding("non-risk", "N2", "Replays are deterministic",
                       f"Three runs of {names(vs)} produce identical output files apart from timing columns "
                       f"({sum(m[v, 'repro_compared_cells'] for v in vs):,.0f} cells compared), including the "
                       "multithreaded GARCH-t refits, the Monte Carlo option revaluation and the multi-start "
                       "CAViaR fits.", ["Rp1"])


def _f_same_pipeline(m: M):
    v = default_variant("ewma")
    if m[v, "baseline_mismatches_same_policy"] == 0:
        return Finding("non-risk", "N3", "The portfolio engines read the market identically on clean data",
                       "With both data protections off, the EWMA(0.94) baseline that EWMA and GARCH-t both compute "
                       "is identical forecast by forecast on the synthetic market: the two pipelines read, align "
                       "and return the market in the same way.", ["Cp1"])


def _f_matched(m: M):
    vs = [v for v in single_defaults() if m[v, "baseline_mismatches"] == 0]
    if vs:
        return Finding("non-risk", "N5", "The model comparison runs on the shared market and on matched targets",
                       f"{names(vs)} reads the market with the repository's shared convention "
                       f"({m.detail(vs[0], 'baseline_mismatches')}). It is the engine in which the research question "
                       "can be answered fairly.", ["Cp1"])


def _f_latency_ok(m: M):
    vs = portfolio_defaults()
    if vs and all(m[v, "refit_max_ms"] < 1000 and m[v, "latency_p99_us"] < 1000 for v in vs):
        single = [v for v in single_defaults() if m.has((v, "model_stage_p50_us"))]
        extra = (f"; {names(single)} updates in {m[single[0], 'model_stage_p50_us'] * 1e3:.0f} ns at the model stage"
                 if single else "")
        return Finding("non-risk", "N4", "Latency and recalibration are far below the bar interval",
                       "Per-bar updates of the portfolio engines take microseconds ("
                       + ", ".join(f"{lab(v)} p99 {m[v, 'latency_p99_us']:.1f} us" for v in vs) + extra
                       + ") and the daily refit at most " + ", ".join(f"{lab(v)} {m[v, 'refit_max_ms']:.0f} ms" for v in
                                                                      defaults() if m.has((v, "refit_max_ms")))
                       + ", against 15-minute bars: at this scale performance does not constrain the designs. "
                       "Timings vary between sessions on the same machine (up to a factor of two), so verdicts near a "
                       "threshold are marked PARTIAL.", ["P1", "P2", "P3"])


def _f_model_cost(m: M):
    v = next(iter(single_defaults()), None)
    if not v:
        return None
    costs = {k.split(":", 1)[1]: x.value for (var, k), x in m.by.items()
             if var == v and k.startswith("model_stage_p50_ns:")}
    states = {k.split(":", 1)[1]: x.value for (var, k), x in m.by.items() if var == v and k.startswith("state_bytes:")}
    if len(costs) < 2 or "Realized-CAViaR" not in costs:
        return None
    cheapest = min(costs, key=costs.get) == "Realized-CAViaR"
    listing = ", ".join(f"{k} {costs[k]:.0f} ns / {states.get(k, math.nan):.0f} B" for k in sorted(costs, key=costs.get))
    if cheapest:
        return Finding("non-risk", "N6", "The main model is the cheapest one under a common boundary",
                       f"In the matched model-stage benchmark ({listing}) Realized-CAViaR has the lowest update cost "
                       "and the smallest state: the computational side of the research question does not penalize "
                       "it.", ["P1", "P4"])
    return Finding("sensitivity", "S-5", "Model-stage cost of the main model",
                   f"In the matched model-stage benchmark ({listing}) Realized-CAViaR is not the cheapest model; the "
                   "difference is a few nanoseconds per update.", ["P1", "P4"])


def _f_window_from_file(m: M):
    tr = {v: m[v, "c1_truncation_changed"] for v in defaults()}
    tw = {v: m[v, "c1_truncation_fixed_window_changed"] for v in defaults()}
    vs = [v for v in defaults() if tr[v] - tw[v] > 0]
    if vs:
        rest = [v for v in vs if tw[v] > 0]
        left = (" (the " + ", ".join(f"{tw[v]:.0f} rows of {lab(v)}" for v in rest) + " that remain are R-2)"
                if rest else "")
        return Finding("risk", "R-1", "The default estimation window depends on how much data follows",
                       f"{names(vs)} derive the default estimation window from the length of the file. Cut "
                       "the feed at T, as a live feed is, and earlier forecasts change ("
                       + ", ".join(f"{lab(v)}: {tr[v] - tw[v]:,.0f} values and rows" for v in vs)
                       + "). With --window-days fixed these changes vanish" + left
                       + ": the window must be configuration, never derived from the input length.", ["C1"])


def _f_end_of_data(m: M):
    vs = [v for v in defaults() if m[v, "c1_truncation_fixed_window_changed"] > 0]
    if vs:
        return Finding("risk", "R-2", "The end of the data is taken as the end of the session",
                       "The replay marks the last bar of the file as a session close (a batch assumption). With a "
                       "fixed window, a feed cut at the first bar "
                       "of a session still changes " + ", ".join(
                           f"{m[v, 'c1_truncation_fixed_window_changed']:.0f} rows in {lab(v)}" for v in vs)
                       + ": the pending overnight forecast is replaced by a record whose target is its own issue "
                       "time. A streaming engine must take the session close from the calendar.", ["C1"])


def _f_dq_policy(m: M):
    v = default_variant("ewma")
    if m[v, "baseline_mismatches"] > 0:
        return Finding("risk", "R-3", "The engines do not apply the same data protection by default",
                       "By default EWMA runs its data-quality layer, GARCH-t its jump filter and Realized-CAViaR "
                       "none, so the shared baseline of the portfolio engines differs in "
                       f"{m[v, 'baseline_mismatches']:.0f} forecasts on the synthetic market and "
                       f"{m[v, 'baseline_mismatches_b3_default']:.0f} on the B3 data (with both protections off: "
                       f"{m[v, 'baseline_mismatches_same_policy']:.0f} and {m[v, 'baseline_mismatches_b3_same_policy']:.0f}). "
                       "Every statistical comparison must state the policy it runs under.", ["Cp1"])


def _f_no_dq_in_comparison(m: M):
    vs = [v for v in single_defaults() if m[v, "degraded_share_dev5_pct"] > 1]
    if vs:
        v = vs[0]
        return Finding("risk", "R-6", "The comparison engine has no data-quality layer",
                       f"On the faulty feed {m[v, 'degraded_share_dev5_pct']:.1f}% of {lab(v)}'s single-stock "
                       f"forecasts move more than 5% ({m.detail(v, 'degraded_share_dev5_pct')}): a bad tick enters "
                       "|R| and RV directly. The garch_t jump filter (data_quality.cpp, causal, one stock at a "
                       "time) could be compiled into realized_caviar like the other garch_t sources it reuses, which "
                       "would protect the single-stock models as well.", ["R1", "R3"])


def _f_boundaries(m: M):
    single, port = single_defaults(), portfolio_defaults()
    if single and port and m.has((single[0], "model_stage_p50_us")):
        g = default_variant("garch_t")
        mc = (f" The GARCH-t Monte Carlo revaluation of the option book ({m[g, 'mc_revaluation_p50_us'] / 1e3:.1f} ms "
              "per bar at the median) and the EWMA confirmation of suspect prices run outside their timers."
              if m.has((g, "mc_revaluation_p50_us")) else "")
        return Finding("risk", "R-7", "Latency is measured at different boundaries",
                       f"{names(port)} time a bar cycle of a 15-stock book (fills, data quality, correlations, VaR); "
                       f"{names(single)} times the model stage of one instrument in batches of 1000, whose "
                       "percentiles are of batch means." + mc + " The numbers are not comparable across engines: "
                       "the cost side of the research question must use the matched model-stage benchmark, where the "
                       "three models share one boundary.", ["P1"])


def _f_no_interface(m: M):
    vs = [v for v in defaults() if m[v, "m1_files_outside_model"] > 2]
    if vs:
        none = all(m[v, "common_model_interface"] == 0 for v in defaults() if m.has((v, "common_model_interface")))
        ok = [v for v in defaults() if v not in vs and m.has((v, "m1_files_outside_model"))]
        tail = (f" {names(ok)} keeps it to {m[ok[0], 'm1_files_outside_model']:.0f} files, its replay orchestrator, "
                "where the models are selected by position." if ok else "")
        return Finding("risk", "R-4", "No common model interface",
                       ("None of the engines has a model interface. " if none else "")
                       + "In " + names(vs) + " the models are named fields of the engine's result, so a new model "
                       "touches " + " and ".join(f"{m[v, 'm1_files_outside_model']:.0f} files of {lab(v)}" for v in vs)
                       + " outside the model layer." + tail + " The monograph's NFR05 (one interface for the three "
                       "models) is not met by the current code.", ["M1"])


def _f_grid(m: M):
    blocked = [v for v in defaults() if m[v, "m2_config_only"] == 0]
    ok = [v for v in defaults() if m[v, "m2_config_only"] == 1]
    if blocked:
        return Finding("risk", "R-5", "The target time grid is reachable in one engine only",
                       f"{names(blocked)} set bars in whole minutes and forecast one bar ahead, so the grid of the "
                       "project (5-second sampling, a 5-minute VaR issued every minute) needs code there"
                       + (f"; {names(ok)} implements it in its high-frequency mode. The shared 15-minute data keeps "
                          "every experiment on 15-minute bars, so the monograph must state which grid it reports."
                          if ok else "."), ["M2"])


def _f_hidden_shock(m: M):
    vs = [v for v in single_defaults() if m[v, "r2_hidden_shocks"] > 0]
    if vs:
        v = vs[0]
        return Finding("risk", "R-8", "A shock followed by a halt never reaches the CAViaR features",
                       f"{lab(v)} builds its features from four consecutive bars of the session and skips incomplete "
                       f"windows. The scripted market crash ({m.detail(v, 'r2_hidden_shocks')}) is followed by a "
                       "two-bar halt, so no window contains it: the forecasts of that session never see the crash, "
                       "while the matched baselines of the same engine, which take every printed bar, do. A halt "
                       "right after a crash is exactly when the VaR matters most.", ["R2"])


def _f_missing_policy(m: M):
    v = default_variant("ewma")
    if m[v, "baseline_mismatches_b3_same_policy"] > 0:
        return Finding("sensitivity", "S-1", "Missing-data policy of the covariance update",
                       "EWMA keeps a missing asset's variance and shrinks its covariances; GARCH-t imputes the "
                       f"missing return from the observed ones. On the B3 data, bars with up to "
                       f"{m[v, 'b3_partial_bar_max_missing']:.0f} {m.detail(v, 'b3_partial_bar_max_missing')} missing "
                       f"make {m[v, 'baseline_mismatches_b3_same_policy']:.0f} forecasts differ (up to "
                       f"{m[v, 'b3_same_policy_max_diff_pct']:.1f}% of the baseline VaR), with both data protections "
                       "off. Real feeds make this a comparability risk the synthetic market cannot show.", ["Cp1"])


def _f_fill_recalibration(m: M):
    v = default_variant("garch_t")
    mx, nf = m[v, "latency_max_us"], m[v, "latency_max_without_fill_bars_us"]
    if mx > 3 * nf:
        return Finding("sensitivity", "S-2", "Tail recalibration on fills runs inside the bar cycle",
                       f"GARCH-t recalibrates its tail model when a fill changes the book, inside the timed bar "
                       f"update: those bars take up to {mx:,.0f} us against a median of {m[v, 'latency_p50_us']:.1f} us "
                       f"(without fill bars the maximum is {nf:.0f} us). Moving it out of the critical path would "
                       "flatten the latency tail.", ["P1"])


def _f_opening_hour(m: M):
    vs = [v for v in single_defaults() if m[v, "evaluated_from_hour"] > 10.25]
    if vs:
        v = vs[0]
        span = re.search(r"\d+:\d+-\d+:\d+", m.detail(v, "evaluated_from_hour"))
        span = span.group(0) if span else f"{m[v, 'evaluated_from_hour']:.2f} h on"
        return Finding("sensitivity", "S-3", "The matched comparison does not cover the opening hour",
                       f"CAViaR needs one hour of returns of the same session, so {lab(v)} scores every model on "
                       f"targets starting {span} only. The most volatile bars of the day (the opening hour) are "
                       "outside the comparison, so its conclusions about the models hold from 11:00 on.", ["Cp1"])


def _f_reuse(m: M):
    vs = [v for v in defaults() if m[v, "reused_sibling_sources"] > 0]
    if vs:
        v = vs[0]
        return Finding("sensitivity", "S-4", "Reuse by compiling the other modules' sources",
                       f"{lab(v)} compiles {m[v, 'reused_sibling_sources']:.0f} source files of ewma and garch_t "
                       "(loader, EWMA, GARCH, backtest). It guarantees the same conventions and estimators in the "
                       "comparison, but couples the modules with no interface or version: a change in ewma or "
                       "garch_t silently changes the comparison engine. atam/run.bat rebuilds it before every "
                       "evaluation, so a rerun after such a change is the safeguard.", ["Cp1", "M1"])


def _f_dq_tradeoff(m: M):
    if m.has(("ewma", "degraded_share_dev5_pct"), ("ewma_nodq", "degraded_share_dev5_pct"), ("ewma", "r2_max_effect_pct")):
        return Finding("trade-off", "T-1", "Data-quality layer: robustness vs reaction to genuine shocks",
                       f"With the layer, {m['ewma', 'degraded_share_dev5_pct']:.2f}% of the faulty-feed forecasts "
                       f"stray more than 5% from the clean-feed ones (without it: "
                       f"{m['ewma_nodq', 'degraded_share_dev5_pct']:.2f}%). The price is paid on genuine moves: at a "
                       f"shock bar the VaR moves up to {m['ewma', 'r2_max_effect_pct']:.1f}% away from the unprotected "
                       f"one, for {m['ewma', 'r2_bars_affected']:.0f} bar, because the suspect stock keeps its last "
                       "price and a winsorized return until the next bar confirms the move. " + _dq_cost_text(m),
                       ["R1", "R2", "P1"])


def _dq_cost_text(m: M) -> str:
    diff, real = latency_cost(m, "ewma", "ewma_nodq")
    note = " Its confirmation step runs before the timer, so part of its cost is not in this figure."
    if real:
        return f"Timed latency cost: {diff:+.2f} us at the median per bar." + note
    return (f"Timed latency cost: none measurable (median per bar {m['ewma', 'latency_p50_us']:.2f} us with the layer, "
            f"{m['ewma_nodq', 'latency_p50_us']:.2f} us without, inside the run-to-run spread)." + note)


def _f_clip_tradeoff(m: M):
    keys = ("garch_t", "garch_t_nojump", "garch_t_zclip6")
    if not all(m.has((k, "degraded_share_dev5_pct"), (k, "shock_min_ratio")) for k in keys if k != "garch_t_nojump"):
        return None
    if not m.has(("garch_t_nojump", "degraded_share_dev5_pct")):
        return None
    r3 = "" if not all(m.has((k, "r3_damaged_pct")) for k in keys) else (
        f" With every recoverable fault together (R3): {m['garch_t_nojump', 'r3_damaged_pct']:.1f}% unprotected, "
        f"{m['garch_t_zclip6', 'r3_damaged_pct']:.1f}% clipped, {m['garch_t', 'r3_damaged_pct']:.1f}% filtered.")
    return Finding("trade-off", "T-2", "GARCH-t: a stateful jump filter against stateless clipping",
                   f"On the faulty feed, {m['garch_t_nojump', 'degraded_share_dev5_pct']:.1f}% of the unprotected "
                   f"GARCH-t forecasts stray more than 5%; clipping shocks at 6 sd leaves "
                   f"{m['garch_t_zclip6', 'degraded_share_dev5_pct']:.1f}% and the jump filter (default) "
                   f"{m['garch_t', 'degraded_share_dev5_pct']:.2f}%." + r3 + " At the genuine shocks of the clean "
                   f"feed, the lowest ratio to the unprotected VaR is {m['garch_t_zclip6', 'shock_min_ratio']:.3f} "
                   f"with clipping and {m['garch_t', 'shock_min_ratio']:.3f} with the filter: the filter delays a move "
                   "confined to one stock by one bar, then lets it in at full size, while clipping shrinks every "
                   "large shock for as long as it is in the variance.", ["R1", "R2", "R3"])


def _f_fault_catalogue(m: M):
    vs = [v for v in defaults() if m.has((v, "r3_damaged_pct"))]
    if not vs:
        return None
    _, tree = read_csv(UTILITY_TREE)
    row = next((r for r in tree if r["metric"] == "r3_damaged_pct"), None)
    if row is None:
        return None
    limit = float(row["threshold"])
    bad = [v for v in vs if m[v, "r3_damaged_pct"] > limit]
    listing = "; ".join(f"{lab(v)}: {m.detail(v, 'r3_damaged_pct')}" for v in vs)
    if bad:
        return Finding("risk", "R-9", "Moderate doses of every recoverable fault still move the VaR",
                       "On faulty copies of both datasets with missing, duplicated, out-of-order and late rows, halts, "
                       "frozen quotes, bad ticks, corrupt values and early-ending sessions together, "
                       f"{names(bad)} stay above {limit:.0f}% of forecasts more than 5% away from the clean run "
                       f"({listing}). "
                       "atam/results/faults/ shows which fault does it, one kind at a time: missing data is the "
                       "residual problem (no protection can recover a print that never arrives).", ["R3"])
    return Finding("non-risk", "N7", "Every engine absorbs the fault catalogue",
                   f"With every recoverable fault together ({listing}), no engine moves more than {limit:.0f}% of its "
                   "forecasts more than 5% away from the clean run.", ["R3"])


def _f_unprinted_bar(m: M):
    vs = [v for v in [default_variant("garch_t")] if v and m.has((v, "r1_off_after_unprinted"), (v, "degraded_share_dev5_pct"))
          and m[v, "degraded_share_dev5_pct"] > 1 and m[v, "r1_share_dev5_other_pct"] <= 1]
    if not vs:
        return None
    v = vs[0]
    return Finding("sensitivity", "S-6", "An unprinted bar widens the next forecast on purpose",
                   f"On the faulty feed {m[v, 'degraded_share_dev5_pct']:.2f}% of the {lab(v)} forecasts move more than "
                   f"5% from the clean-feed ones, but {m[v, 'r1_off_after_unprinted']:.0f} of them are issued at or "
                   "right after a bar in which a stock did not print; without those the share is "
                   f"{m[v, 'r1_share_dev5_other_pct']:.2f}%. The engine marks a stock that skips a bar at its last "
                   "print, so the P&L of its next bar spans two bars, and it adds the variance of the unprinted bar "
                   "to that forecast: the clean-feed VaR covers one bar, the faulty-feed one covers the P&L it is "
                   "actually scored on. R1 counts this deliberate response as a deviation; the threshold would be "
                   "met if the scenario compared forecasts of the same P&L horizon.", ["R1"])


def _f_book_cache(m: M):
    vs = [v for v in [default_variant("garch_t")] if v and m.has((v, "latency_p99_us_no_book"), (v, "latency_p99_us"))
          and m[v, "latency_p99_us"] > 2 * m[v, "latency_p99_us_no_book"]]
    if not vs:
        return None
    g = default_variant("garch_t")
    mc = (f" (the revaluation itself takes {m[g, 'mc_revaluation_p50_us'] / 1e3:.1f} ms per bar at the median)"
          if m.has((g, "mc_revaluation_p50_us")) else "")
    return Finding("sensitivity", "S-7", "The option revaluation slows down the bar update it is not part of",
                   "With the option book, the Monte Carlo revaluation runs on the same thread between two bar updates"
                   + mc + " and evicts the engine's data from the processor caches, so the next update starts cold: "
                   + "; ".join(f"{lab(v)} p99 {m[v, 'latency_p99_us']:.1f} us with the book, "
                               f"{m[v, 'latency_p99_us_no_book']:.1f} us without it (p50 {m[v, 'latency_p50_us']:.1f} "
                               f"vs {m[v, 'latency_p50_us_no_book']:.1f} us)" for v in vs)
                   + ". Running the revaluation on another core after the VaR is published would keep the bar update "
                   "at its own cost.", ["P1"])


FINDING_RULES = [_f_hidden_shock, _f_no_value_lookahead, _f_deterministic, _f_same_pipeline, _f_matched, _f_latency_ok, _f_model_cost,
                 _f_window_from_file, _f_end_of_data, _f_dq_policy, _f_no_interface, _f_grid, _f_no_dq_in_comparison,
                 _f_boundaries, _f_missing_policy, _f_fill_recalibration, _f_opening_hour, _f_reuse, _f_dq_tradeoff,
                 _f_clip_tradeoff, _f_fault_catalogue, _f_unprinted_bar, _f_book_cache]


# ----------------------------------------------------------------------------- trade-off map


def tradeoff_rows(m: M) -> tuple[list[str], list[tuple[str, str, dict]]]:
    """Architectural decisions x quality attributes: (sign, short evidence) per affected attribute."""
    attrs = ["Robustness\n(R1, R3)", "Shock reaction\n(R2)", "Comparability\n(Cp1)", "Correctness\n(C1)", "Latency\n(P1)",
             "Scope / scale\n(P2)", "Modifiability\n(M1, M2)"]
    R1, R2, CP, C1, P1, P2, MO = attrs
    g, e, c = default_variant("garch_t"), default_variant("ewma"), default_variant("realized_caviar")
    cost, real = latency_cost(m, "ewma", "ewma_nodq")
    dq_cost_cell = f"{cost:+.1f} us median" if real else "no cost above\nrun-to-run noise"
    rows = [
        ("Data-quality layer, suspect move\nconfirmed on the next bar", "EWMA", {
            R1: ("+", f"{m['ewma_nodq', 'degraded_share_dev5_pct']:.1f}% -> {m['ewma', 'degraded_share_dev5_pct']:.2f}%\noff >5%"),
            R2: ("~", f"{m['ewma', 'r2_max_effect_pct']:.1f}% off for\n{m['ewma', 'r2_bars_affected']:.0f} bar"),
            CP: ("-", f"{m[e, 'baseline_mismatches']:.0f} / {m[e, 'baseline_mismatches_b3_default']:.0f} forecasts\ndiffer"),
            P1: ("~", dq_cost_cell)}),
        ("Jump filter: isolated jump held\none bar (default)", "GARCH-t", {
            R1: ("+", f"{m['garch_t_nojump', 'degraded_share_dev5_pct']:.1f}% -> {m[g, 'degraded_share_dev5_pct']:.2f}%\noff >5%"),
            R2: ("~", f"x{m[g, 'shock_min_ratio']:.3f}, {m[g, 'r2_bars_affected']:.0f}\nbar(s) affected"),
            CP: ("-", "a policy of its\nown (R-3)")}),
        ("Stateless clipping of shocks\nat 6 sd (option)", "GARCH-t", {
            R1: ("+", f"{m['garch_t_nojump', 'degraded_share_dev5_pct']:.1f}% -> {m['garch_t_zclip6', 'degraded_share_dev5_pct']:.1f}%\noff >5%"),
            R2: ("-", f"x{m['garch_t_zclip6', 'shock_min_ratio']:.3f}, {m['garch_t_zclip6', 'r2_bars_affected']:.0f}\nbars affected")}),
        ("No data-quality layer\n(loader only)", "Realized-CAViaR", {
            R1: ("-", f"{m[c, 'degraded_share_dev5_pct']:.1f}% of stock\nforecasts off >5%"),
            R2: ("+", "real moves enter\nunfiltered")}),
        ("Missing asset in a bar: keep\nvariance vs impute", "EWMA, GARCH-t", {
            CP: ("-", f"{m[e, 'baseline_mismatches_b3_same_policy']:.0f} B3 forecasts\ndiffer")}),
        ("One instrument per run,\nthree matched models", "Realized-CAViaR", {
            CP: (("+", "returns = shared\nconvention") if m[c, "baseline_mismatches"] == 0
                 else ("-", f"{m[c, 'baseline_mismatches']:.0f} targets off the\nshared convention")),
            P2: ("-", "no portfolio VaR;\nN runs for N stocks")}),
        ("Features: four consecutive\nbars of the session", "Realized-CAViaR", {
            CP: ("~", "opening hour\nnot evaluated"),
            **({R2: ("-", f"{m[c, 'r2_hidden_shocks']:.0f} shock before a halt\nnever enters")}
               if m[c, "r2_hidden_shocks"] > 0 else {})}),
    ]
    # decisions that the evidence of this run still shows (a fixed one drops out of the map)
    if sum(m[v, "c1_truncation_changed"] - m[v, "c1_truncation_fixed_window_changed"] for v in defaults()) > 0:
        rows.append(("Window derived from the\nlength of the file", "engines", {
            C1: ("-", f"{sum(m[v, 'c1_truncation_changed'] for v in defaults()):,.0f} values\nchange when cut")}))
    if any(m[v, "c1_truncation_fixed_window_changed"] > 0 for v in defaults()):
        rows.append(("End of data = end of session", "replays", {
            C1: ("-", f"{max(m[v, 'c1_truncation_fixed_window_changed'] for v in defaults()):.0f} rows replaced\nwhen cut")}))
    if m[g, "latency_max_us"] > 3 * m[g, "latency_max_without_fill_bars_us"]:
        rows.append(("Tail recalibration on fills\ninside the bar cycle", "GARCH-t", {
            P1: ("-", f"up to {m[g, 'latency_max_us']:,.0f} us\nvs {m[g, 'latency_p50_us']:.0f} us median")}))
    if m.has((g, "latency_p99_us_no_book")) and m[g, "latency_p99_us"] > 2 * m[g, "latency_p99_us_no_book"]:
        rows.append(("Option revaluation on the\nbar-update thread", "GARCH-t", {
            P1: ("-", f"p99 {m[g, 'latency_p99_us']:.1f} us vs\n{m[g, 'latency_p99_us_no_book']:.1f} us without")}))
    if m.has((g, "r1_off_after_unprinted")) and m[g, "r1_off_after_unprinted"] > 0:
        rows.append(("Unprinted bar: its variance\nadded to the next forecast", "GARCH-t", {
            R1: ("~", f"{m[g, 'r1_off_after_unprinted']:.0f} forecasts off;\n"
                      f"{m[g, 'r1_share_dev5_other_pct']:.2f}% otherwise")}))
    rows += [
        ("Compiles ewma and garch_t\nsources", "Realized-CAViaR", {
            CP: ("+", "same loader and\nestimators"),
            MO: ("-", f"{m[c, 'reused_sibling_sources']:.0f} foreign files,\nsilent coupling")}),
        ("Models as named fields /\nselected by position", "all three", {
            MO: ("-", f"{m[e, 'm1_files_outside_model']:.0f} / {m[g, 'm1_files_outside_model']:.0f} / "
                      f"{m[c, 'm1_files_outside_model']:.0f} files\nto add a model")}),
        ("Whole-minute bars, one-bar\nhorizon", "EWMA, GARCH-t", {
            MO: ("-", "5 s / 1 min / 5 min\nneeds code")}),
    ]
    return attrs, rows


# ----------------------------------------------------------------------------- figures


def fig_utility_tree(tree: list[dict]) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from plotstyle import AXIS, GRID, INK, INK_2, MUTED, SLIDE, SURFACE, heading, save

    fig = plt.figure(figsize=SLIDE)
    heading(fig, "Utility tree: the quality ruler the three architectures are measured against",
            "Business drivers -> quality attributes -> scenarios with a measurable response.\n"
            "(Importance, Difficulty): H = high, M = medium, L = low; bold = (H,H) or (H,M), analysed first")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(tree)
    ys = [0.96 - i * (0.92 / (n - 1)) for i in range(n)]
    attrs = list(dict.fromkeys(r["attribute"] for r in tree))
    ay = {a: sum(y for r, y in zip(tree, ys) if r["attribute"] == a) / sum(r["attribute"] == a for r in tree)
          for a in attrs}

    def box(x, y, w, h, text, weight="normal", fc=SURFACE, size=10.5):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                    fc=fc, ec=AXIS, lw=0.8))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=size, fontweight=weight, color=INK)

    box(0.015, 0.5, 0.085, 0.09, "Utility", "bold", GRID, 11.5)
    for a in attrs:
        ax.plot([0.10, 0.115, 0.115, 0.13], [0.5, 0.5, ay[a], ay[a]], color=MUTED, lw=0.8)
        box(0.13, ay[a], 0.135, 0.06, a, "bold", GRID, 10)
    for r, y in zip(tree, ys):
        ax.plot([0.265, 0.28, 0.28, 0.29], [ay[r["attribute"]], ay[r["attribute"]], y, y], color=MUTED, lw=0.8)
        hot = r["importance"] == "H" and r["difficulty"] in ("H", "M")
        thr = {"pending": "pending", "report": "reported"}.get(r["op"], f"{r['op']} {r['threshold']} {r['unit']}")
        ax.text(0.295, y + 0.012, f"{r['id']}  ({r['importance']},{r['difficulty']})  {r['refinement']}",
                ha="left", va="center", fontsize=10.5, fontweight="bold" if hot else "normal", color=INK)
        ax.text(0.295, y - 0.022, f"{r['response_measure']}: {thr}", ha="left", va="center", fontsize=8.8,
                color=INK_2)
    return save(fig, "atam_utility_tree")


def fig_scorecard(tree: list[dict], cells: dict) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from plotstyle import INK, INK_2, SLIDE, STATUS_COLOR, STATUS_MARK, heading, save

    fig = plt.figure(figsize=SLIDE)
    met = {e: sum(cells[(r["id"], e)][1] == "MET" for r in tree) for e in ENGINE_ORDER}
    judged = {e: sum(cells[(r["id"], e)][1] in ("MET", "NOT MET", "PARTIAL") for r in tree) for e in ENGINE_ORDER}
    partial = {e: sum(cells[(r["id"], e)][1] == "PARTIAL" for r in tree) for e in ENGINE_ORDER}
    heading(fig, "Scorecard: each architecture against the quality ruler",
            "Value measured on this machine against the threshold of the utility tree. "
            + "; ".join(f"{ENGINES[e].label} meets {met[e]} of {judged[e]}"
                        + (f" ({partial[e]} borderline)" if partial[e] else "") for e in ENGINE_ORDER if judged[e])
            + ".\nP1 boundaries differ: the portfolio engines time a 15-stock bar cycle, Realized-CAViaR the model "
            "stage of one instrument (R-7); its R1 unit is a single-stock forecast")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.845])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x_cols = [0.43, 0.62, 0.81]
    w = 0.175
    h = 0.90 / (len(tree) + 1)
    for j, ek in enumerate(ENGINE_ORDER):
        ax.text(x_cols[j] + w / 2, 0.965, ENGINES[ek].label, ha="center", va="center", fontsize=12,
                fontweight="bold", color=INK)
    for i, r in enumerate(tree):
        y = 0.965 - (i + 1) * h
        thr = {"pending": "pending", "report": "reported"}.get(r["op"], f"{r['op']} {r['threshold']} {r['unit']}")
        ax.text(0.015, y + 0.012, f"{r['id']}  {r['attribute']}: {r['refinement']}", ha="left", va="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(0.015, y - 0.019, f"({r['importance']},{r['difficulty']})  {r['metric']} {thr}", ha="left",
                va="center", fontsize=8.5, color=INK_2)
        for j, ek in enumerate(ENGINE_ORDER):
            value, st = cells[(r["id"], ek)]
            ax.add_patch(FancyBboxPatch((x_cols[j], y - h * 0.40), w, h * 0.80,
                                        boxstyle="round,pad=0,rounding_size=0.006", fc=STATUS_COLOR[st], ec="none"))
            strong = st in ("MET", "NOT MET")  # white on green / red; PARTIAL keeps dark ink on yellow
            ink = "#ffffff" if strong else INK
            label = st + (f"   {value}" if st in ("MET", "NOT MET", "REPORTED", "PARTIAL") else "")
            ax.text(x_cols[j] + 0.008, y, STATUS_MARK[st], ha="left", va="center", fontsize=10.5, color=ink,
                    fontfamily="DejaVu Sans", fontweight="bold")  # the UI font has no check / ballot glyphs
            ax.text(x_cols[j] + 0.024, y, label, ha="left", va="center", fontsize=9.5, color=ink,
                    fontweight="bold" if strong else "normal")
    return save(fig, "atam_scorecard")


def fig_tradeoffs(m: M) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from plotstyle import GRID, INK, INK_2, SLIDE, heading, save

    attrs, rows = tradeoff_rows(m)
    fig = plt.figure(figsize=SLIDE)
    heading(fig, "Sensitivity and trade-off points: which decision moves which quality attribute",
            "Each row is an architectural decision found in the engines; + improves the attribute, - degrades it, "
            "~ small effect. Numbers are the measured evidence")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.875])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x0 = 0.245
    cw = (1 - x0 - 0.008) / len(attrs)
    rh = 0.915 / (len(rows) + 1)
    fill = {"+": "#d6efd6", "-": "#f6d5d5", "~": "#fdeccb"}
    for j, a in enumerate(attrs):
        ax.text(x0 + (j + 0.5) * cw, 0.962, a, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    for i, (decision, where, effects) in enumerate(rows):
        y = 0.958 - (i + 1) * rh
        ax.text(0.010, y + 0.004, decision, ha="left", va="center", fontsize=8.8, color=INK, fontweight="bold",
                linespacing=1.05)
        ax.text(0.238, y, where, ha="right", va="center", fontsize=7.8, color=INK_2)
        for j, a in enumerate(attrs):
            x = x0 + j * cw
            sign, text = effects.get(a, ("", ""))
            ax.add_patch(FancyBboxPatch((x + 0.002, y - rh * 0.45), cw - 0.004, rh * 0.90,
                                        boxstyle="round,pad=0,rounding_size=0.005", fc=fill.get(sign, GRID),
                                        ec="none", alpha=1.0 if sign else 0.30))
            if sign:
                ax.text(x + 0.008, y, sign, ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)
                ax.text(x + 0.022, y, text, ha="left", va="center", fontsize=7.8, color=INK, linespacing=1.05)
    return save(fig, "atam_tradeoffs")


# ----------------------------------------------------------------------------- LaTeX


def _tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("<=", r"$\leq$"), (">=", r"$\geq$"), ("->", r"$\rightarrow$")):
        s = s.replace(a, b)
    return s


def _thr(r: dict) -> str:
    return {"pending": "pending", "report": "reported"}.get(r["op"], f"{r['op']} {r['threshold']} {r['unit']}")


def write_latex(tree: list[dict], cells: dict, findings: list[Finding]) -> None:
    out = RESULTS / "latex"
    out.mkdir(parents=True, exist_ok=True)
    lines = [r"% generated by atam/source/part_scorecard.py - do not edit by hand", r"\begin{quadro}[htbp]",
             r"\caption{\label{quadro:utility-tree}Utility tree of the intraday risk engines (ATAM)}", r"\footnotesize",
             r"\begin{tabular}{|p{0.6cm}|p{2.2cm}|p{5.6cm}|p{0.6cm}|p{4.2cm}|}", r"\hline",
             r"\textbf{ID} & \textbf{Attribute} & \textbf{Scenario (stimulus $\rightarrow$ response)} & "
             r"\textbf{I,D} & \textbf{Response measure} \\ \hline"]
    for r in tree:
        lines.append(f"{r['id']} & {_tex(r['attribute'])} & {_tex(r['stimulus'])} $\\rightarrow$ {_tex(r['response'])} & "
                     f"{r['importance']},{r['difficulty']} & {_tex(r['response_measure'])}: {_tex(_thr(r))} \\\\ \\hline")
    lines += [r"\end{tabular}", r"\fonte{Prepared by the project team.}", r"\end{quadro}"]
    (out / "utility_tree.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"% generated by atam/source/part_scorecard.py - do not edit by hand", r"\begin{quadro}[htbp]",
             r"\caption{\label{quadro:scorecard}ATAM scorecard: measured response of each engine}", r"\footnotesize",
             r"\begin{tabular}{|p{0.6cm}|p{3.0cm}|p{3.4cm}|p{3.4cm}|p{3.4cm}|}", r"\hline",
             r"\textbf{ID} & \textbf{Threshold} & \textbf{" + "} & \\textbf{".join(ENGINES[e].label for e in ENGINE_ORDER)
             + r"} \\ \hline"]
    for r in tree:
        row = [r["id"], _tex(_thr(r))]
        for e in ENGINE_ORDER:
            value, st = cells[(r["id"], e)]
            row.append(_tex(f"{st} ({value})" if st in ("MET", "NOT MET", "REPORTED", "PARTIAL") else st.lower()))
        lines.append(" & ".join(row) + r" \\ \hline")
    lines += [r"\end{tabular}", r"\fonte{Measured by atam/run.bat on the machine in atam/results/environment.txt.}",
              r"\end{quadro}"]
    (out / "scorecard.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"% generated by atam/source/part_scorecard.py - do not edit by hand"]
    for kind, title in (("risk", "Risks"), ("sensitivity", "Sensitivity points"), ("trade-off", "Trade-off points"),
                        ("non-risk", "Non-risks")):
        items = [f for f in findings if f.kind == kind]
        if not items:
            continue
        lines += [r"\paragraph{" + title + "}", r"\begin{itemize}"]
        lines += [r"\item \textbf{" + f"{f.id} {_tex(f.title)}" + r".} " + _tex(f.text) for f in items]
        lines.append(r"\end{itemize}")
    (out / "findings.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- part


def _wrap(text: str, width: int = 104) -> list[str]:
    out, line = [], ""
    for w in text.split():
        if line and len(line) + len(w) > width:
            out.append(line.rstrip())
            line = ""
        line += w + " "
    return out + [line.rstrip()]


def run(fresh: bool = False):
    _, tree = read_csv(UTILITY_TREE)
    m = M()
    cells, rows = {}, []
    for r in tree:
        row = {k: r[k] for k in ("id", "attribute", "refinement", "importance", "difficulty", "metric", "op",
                                 "threshold", "unit")}
        for ek in ENGINE_ORDER:
            v = default_variant(ek) if ENGINES[ek].enabled else None
            value = m[v, r["metric"]] if v else math.nan
            unit = m.unit(v, r["metric"]) if v else ""
            st = status(value, r["op"], r["threshold"], unit, m[v, r["metric"] + "_min"] if v else math.nan,
                        m[v, r["metric"] + "_max"] if v else math.nan)
            if st == "PARTIAL":
                detail_extra = (f"borderline: the threshold lies within the run-to-run range "
                                f"{m[v, r['metric'] + '_min']:.2f}-{m[v, r['metric'] + '_max']:.2f} (x{NOISE_MARGIN} margin)")
            else:
                detail_extra = ""
            cells[(r["id"], ek)] = (fmt(value, unit or r["unit"]), st)
            row[f"{ek}_value"] = "" if math.isnan(value) else value
            row[f"{ek}_status"] = st
            row[f"{ek}_detail"] = "; ".join(x for x in (m.detail(v, r["metric"]) if v else ENGINES[ek].note,
                                                        detail_extra) if x)
        rows.append(row)
    write_csv(RESULTS / "scorecard.csv", list(rows[0].keys()), rows)

    findings = [f for f in (rule(m) for rule in FINDING_RULES) if f]
    write_csv(RESULTS / "findings.csv", ["kind", "id", "title", "text", "scenarios"],
              [[f.kind, f.id, f.title, f.text, " ".join(f.scenarios)] for f in findings])

    figs = [fig_utility_tree(tree), fig_scorecard(tree, cells), fig_tradeoffs(m)]
    write_latex(tree, cells, findings)

    env = (RESULTS / "environment.txt").read_text(encoding="utf-8") if (RESULTS / "environment.txt").is_file() else ""
    head = Section("ATAM evaluation of the intraday risk engines")
    head.add("Architectures: ewma and garch_t (portfolio engines) and realized_caviar (single-instrument engine with")
    head.add("matched EWMA-N and GARCH-t baselines), replaying the same data. Quality ruler: atam/utility_tree.csv.")
    head.add("Every number below is reproduced by atam/run.bat (Windows) or atam/run.sh.")
    sc = Section("Scorecard (default configuration of each engine)")
    sc.add(f"{'id':<4} {'(I,D)':<6} {'threshold':<20} " + "".join(f"{ENGINES[e].label:<26}" for e in ENGINE_ORDER))
    for r in tree:
        sc.add(f"{r['id']:<4} ({r['importance']},{r['difficulty']})  {_thr(r):<20} " + "".join(
            f"{(cells[(r['id'], e)][1] + ' ' + (cells[(r['id'], e)][0] if cells[(r['id'], e)][1] in ('MET', 'NOT MET', 'REPORTED', 'PARTIAL') else '')):<26}"
            for e in ENGINE_ORDER))
    fs = Section("Findings")
    for kind in ("risk", "sensitivity", "trade-off", "non-risk"):
        for f in (f for f in findings if f.kind == kind):
            fs.add(f"[{f.kind}] {f.id} {f.title} ({', '.join(f.scenarios)})")
            for line in _wrap(f.text):
                fs.add("    " + line)
    fs.add("")
    fs.add("figures: " + ", ".join(figs))
    parts = [(RESULTS / p / "report.txt").read_text(encoding="utf-8") for p in PARTS if (RESULTS / p / "report.txt").is_file()]
    text = "\n\n".join([head.text(), env.strip(), sc.text(), fs.text(), *[p.strip() for p in parts]])
    write_text(RESULTS / "report.txt", text)
    print("   " + "\n   ".join((sc.text() + "\n" + fs.text()).splitlines()))
    return None
