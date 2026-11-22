"""The architectures under evaluation: one engine per VaR technique, each with its configurations.

An *engine* is one of the repository's modules (ewma, garch_t, realized_caviar). A *variant* is one
configuration of an engine; the default variant is the engine as designed, and the other variants
switch a single architectural decision on or off (e.g. the data-quality layer) so that its effect
can be measured. Every run is cached in atam/work/runs/<variant>/<tag>/ and reused while the
command line, the executable and the input files stay the same: rebuilding an engine invalidates
its runs automatically.

Everything the evaluation assumes about an engine (program, output files, columns, report text)
is declared in ENGINES below. When an engine changes its interface, edit its entry here; the runs
fail early with the name of the missing column or file instead of producing wrong evidence.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from common import REPO, SYNTHETIC, WORK, read_csv, rel, sha256, write_csv


@dataclass(frozen=True)
class Engine:
    key: str
    label: str
    folder: str  # module folder in the repository
    program: str  # replay program
    model: str  # main VaR model of the engine
    var99: str  # main-model VaR 99% column of the forecast file
    unprotected: str  # variant with the engine's data-quality protection switched off
    benchmark: str = ""  # separate benchmark program (empty: timings come from the replay output)
    baseline_sigma: str = ""  # shared EWMA(0.94) baseline columns (portfolio engines only)
    baseline_var99: str = ""
    exe_dirs: tuple[str, ...] = ("bin",)  # where the programs are looked for, in order
    risk_file: str = "risk_15m.csv"  # forecast file in the shared schema (written by an adapter if needed)
    forecast_files: tuple[str, ...] = ("risk_15m.csv", "risk_overnight.csv")  # compared by C1
    calibration_files: tuple[tuple[str, tuple[str, ...], str, str], ...] = ()  # (file, key, time col, kind)
    per_ticker: bool = False  # one instrument per run (--ticker)
    default_ticker: str = ""
    window_pattern: str = r"window (\d+) days"  # estimation window as printed in report.txt
    adapter: str = ""  # function in ADAPTERS that writes risk_file from the engine's own output
    enabled: bool = True
    note: str = ""

    def exe(self, program: str | None = None) -> Path:
        name = program or self.program
        for d in self.exe_dirs:
            for candidate in (REPO / self.folder / d / f"{name}.exe", REPO / self.folder / d / name):
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"{self.folder}: program {name} not found in {', '.join(self.exe_dirs)}; build the "
                                f"{self.folder} module first")


@dataclass(frozen=True)
class Variant:
    key: str
    engine: str
    label: str  # short label for tables and figures
    args: tuple[str, ...]
    default: bool
    decision: str  # architectural decision this variant isolates


ENGINES: dict[str, Engine] = {
    "ewma": Engine(
        key="ewma", label="EWMA", folder="ewma", program="ewma_risk", benchmark="benchmark",
        model="EWMA-Normal (estimated lambda)", var99="n_var_99", unprotected="ewma_nodq",
        baseline_sigma="ref_sigma", baseline_var99="ref_var_99",
        calibration_files=(("daily_summary.csv", ("date",), "date", "date"),),
    ),
    "garch_t": Engine(
        key="garch_t", label="GARCH-t", folder="garch_t", program="intraday_risk", benchmark="benchmark",
        model="GARCH(1,1)-t", var99="var_99", unprotected="garch_t_nojump",
        baseline_sigma="ewma_sigma", baseline_var99="ewma_var_99",
        calibration_files=(("daily_summary.csv", ("date",), "date", "date"),
                           ("garch_params_daily.csv", ("date", "ticker"), "date", "date")),
    ),
    "realized_caviar": Engine(
        key="realized_caviar", label="Realized-CAViaR", folder="realized_caviar", program="realized_caviar",
        model="Realized-CAViaR (matched EWMA-N and GARCH-t baselines)", var99="var_99",
        unprotected="realized_caviar", exe_dirs=("bin", "build", "build/Release"), risk_file="atam_risk.csv",
        forecast_files=("atam_risk.csv",),
        calibration_files=(("params_daily.csv", ("effective_time", "model", "phase"), "effective_time", "epoch"),),
        per_ticker=True, default_ticker="XPET3", window_pattern=r"initial training sessions=(\d+)",
        adapter="caviar",
        note="one instrument per run; built from source by realized_caviar/build.bat (no versioned binary)",
    ),
}

VARIANTS: dict[str, Variant] = {
    "ewma": Variant("ewma", "ewma", "EWMA (DQ on)", (), True,
                    "data-quality layer with SPIKE rule confirmed on the next bar (default)"),
    "ewma_nodq": Variant("ewma_nodq", "ewma", "EWMA (DQ off)", ("--no-dq",), False,
                         "data-quality layer switched off (plain loader)"),
    "garch_t": Variant("garch_t", "garch_t", "GARCH-t (filter on)", (), True,
                       "jump filter: an isolated price jump is held one bar, dropped if it reverts, "
                       "kept one bar late otherwise (default)"),
    "garch_t_nojump": Variant("garch_t_nojump", "garch_t", "GARCH-t (filter off)", ("--no-jump-filter",), False,
                              "jump filter switched off: every print enters the model"),
    "garch_t_zclip6": Variant("garch_t_zclip6", "garch_t", "GARCH-t (clip 6 sd)", ("--no-jump-filter", "--z-clip", "6"),
                              False, "stateless clipping of standardized shocks at 6 sd instead of the jump filter"),
    "realized_caviar": Variant("realized_caviar", "realized_caviar", "Realized-CAViaR", (), True,
                               "no data-quality layer; one instrument, three matched models (default)"),
}


# ----------------------------------------------------------------------------- adapters


def epoch_to_iso(v: str) -> str:
    """Exchange-local wall-clock epoch seconds (realized_caviar) -> 'YYYY-MM-DD HH:MM:SS'."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    return dt.datetime.fromtimestamp(x, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


CAVIAR_MODELS = {"Realized-CAViaR": "", "EWMA-N": "ewma_", "EWMA-N (ref)": "ewma_", "GARCH-t": "garch_",
                 "EWMA-t": "ewmat_"}
CAVIAR_COLUMNS = ["forecast_time", "target_time", "net_exposure", "var_99", "quantile", "ewma_var_99",
                  "ewma_quantile", "garch_var_99", "garch_quantile", "ewmat_var_99", "ewmat_quantile",
                  "target_return", "realized_pnl", "evaluated", "update_us"]


def caviar_adapter(out: Path) -> None:
    """forecasts.csv (one row per model and forecast) -> atam_risk.csv (one row per forecast, ISO times)."""
    header, rows = read_csv(out / "forecasts.csv")
    need = {"model", "forecast_time", "target_time", "quantile", "exposure", "var", "target_return", "realized_pnl",
            "evaluated"}
    if need - set(header):
        raise RuntimeError(f"realized_caviar: forecasts.csv has no column {', '.join(sorted(need - set(header)))}; "
                           "update caviar_adapter in atam/source/engines.py")
    _, risk = read_csv(out / "risk_15m.csv")
    confs = {r["confidence"] for r in risk}
    if any(abs(float(c) - 0.99) > 1e-9 for c in confs):
        raise RuntimeError(f"realized_caviar ran at confidence {confs}, the evaluation reads VaR 99%")
    by_time: dict[str, dict] = {}
    for r in rows:
        prefix = CAVIAR_MODELS.get(r["model"])
        if prefix is None:
            raise RuntimeError(f"realized_caviar: unknown model {r['model']!r}; add it to CAVIAR_MODELS in engines.py")
        t = epoch_to_iso(r["forecast_time"])
        row = by_time.setdefault(t, {"forecast_time": t, "update_us": ""})
        row[f"{prefix}var_99"] = r["var"]
        row[f"{prefix}quantile"] = r["quantile"]
        if prefix == "":  # target and outcome are shared by construction; taken from the main model
            row.update({"target_time": epoch_to_iso(r["target_time"]), "net_exposure": r["exposure"],
                        "target_return": r["target_return"], "realized_pnl": r["realized_pnl"],
                        "evaluated": r["evaluated"]})
    write_csv(out / "atam_risk.csv", CAVIAR_COLUMNS, [by_time[t] for t in sorted(by_time)])


ADAPTERS = {"caviar": caviar_adapter}


# ----------------------------------------------------------------------------- runs


def default_variants() -> list[Variant]:
    return [v for v in VARIANTS.values() if v.default and ENGINES[v.engine].enabled]


def enabled_engines() -> list[Engine]:
    return [e for e in ENGINES.values() if e.enabled]


COMMON_COLUMNS = ("forecast_time", "target_time", "net_exposure", "realized_pnl", "update_us")


def check_outputs(e: Engine, out: Path) -> None:
    """Fails early, with the line to edit, when an engine's output no longer has a column the evaluation reads."""
    header, _ = read_csv(out / e.risk_file)
    needed = [*COMMON_COLUMNS, e.var99, e.baseline_sigma, e.baseline_var99]
    missing = [c for c in needed if c and c not in header]
    if missing:
        raise RuntimeError(
            f"{e.folder}: {e.risk_file} has no column {', '.join(missing)}. If the engine renamed it, update "
            f"ENGINES['{e.key}'] (or COMMON_COLUMNS) in atam/source/engines.py. Columns now: {', '.join(header)}")


def _inputs_digest(data_dir: Path, prices: Path | None) -> dict[str, str]:
    files = sorted(p for p in data_dir.iterdir() if p.is_file())
    if prices is not None:
        files.append(prices)
    return {rel(p): sha256(p) for p in files}


_REFRESHED: set[Path] = set()  # run folders already rerun by this process under --fresh
MACHINE = f"{platform.node()}|{platform.processor()}|{os.cpu_count()}"


def run(variant: str, tag: str, data_dir: Path, prices: Path | None = None, fresh: bool = False,
        extra: tuple[str, ...] = (), ticker: str | None = None) -> Path:
    """Runs one engine variant on a dataset and returns its output folder (cached).

    extra: further command-line options of this run only (e.g. a fixed estimation window).
    ticker: instrument of a single-instrument engine (default: the engine's default_ticker)."""
    v = VARIANTS[variant]
    e = ENGINES[v.engine]
    exe = e.exe()
    args_t: list[str] = []
    if e.per_ticker:
        ticker = ticker or e.default_ticker
        args_t = ["--ticker", ticker]
        tag = f"{tag}_{ticker}"
    out = WORK / "runs" / v.key / tag
    args = ["--data", str(data_dir), "--out", str(out), *args_t, *v.args, *extra]
    if prices is not None:
        args += ["--prices", str(prices)]
    manifest = {
        "variant": v.key,
        "args": [rel(Path(a)) if os.path.isabs(a) else a for a in args],
        "exe": rel(exe),
        "exe_sha256": sha256(exe),
        "inputs": _inputs_digest(data_dir, prices),
        "machine": MACHINE,  # timings from another machine are never reused
    }
    stamp = out / "atam_run.json"
    if (not fresh or out in _REFRESHED) and stamp.is_file():  # --fresh reruns each folder once per process
        old = json.loads(stamp.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in manifest} == manifest and (out / e.risk_file).is_file():
            return out
    _REFRESHED.add(out)
    if out.exists():
        for p in out.iterdir():
            if p.is_file():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run([str(exe), *args], cwd=REPO / e.folder, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - t0
    (out / "atam_run.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not (out / "report.txt").is_file():
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise RuntimeError(f"{v.key} failed (exit {proc.returncode}) on {rel(data_dir)}:\n{tail}")
    if e.adapter:
        ADAPTERS[e.adapter](out)
    check_outputs(e, out)
    manifest["seconds"] = round(seconds, 2)
    stamp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"    ran {v.key:<15} {tag:<34} {seconds:6.1f} s")
    return out


def run_benchmark(engine: str, tag: str, fresh: bool = False) -> str:
    """Runs an engine's benchmark program and returns its text output (cached)."""
    e = ENGINES[engine]
    exe = e.exe(e.benchmark)
    out = WORK / "bench" / e.key / f"{tag}.txt"
    stamp = out.with_suffix(".json")
    digest = sha256(exe)
    if ((not fresh or out in _REFRESHED) and out.is_file() and stamp.is_file()
            and json.loads(stamp.read_text()).get("exe_sha256") == digest
            and json.loads(stamp.read_text()).get("machine") == MACHINE):
        return out.read_text(encoding="utf-8")
    _REFRESHED.add(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run([str(exe)], cwd=REPO / e.folder, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{e.key} benchmark failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    out.write_text(proc.stdout, encoding="utf-8")
    stamp.write_text(json.dumps({"exe": rel(exe), "exe_sha256": digest, "machine": MACHINE,
                                 "seconds": round(time.perf_counter() - t0, 2)}), encoding="utf-8")
    print(f"    ran {e.key:<15} benchmark {tag}")
    return proc.stdout


REPS = 3  # repeated clean-feed replays (reproducibility and latency)


def clean_runs(variant: str, fresh: bool = False, reps: int = REPS, ticker: str | None = None) -> list[Path]:
    """Replays of the clean synthetic feed shared by every part (tags synthetic_rep1..repN)."""
    return [run(variant, f"synthetic_rep{i}", SYNTHETIC, fresh=fresh, ticker=ticker) for i in range(1, reps + 1)]


def engine_of(out: Path) -> Engine:
    """The engine that produced a run folder (from its manifest)."""
    variant = json.loads((out / "atam_run.json").read_text(encoding="utf-8"))["variant"]
    return ENGINES[VARIANTS[variant].engine]


def risk_path(out: Path) -> Path:
    """The forecast file of a run in the shared schema."""
    return out / engine_of(out).risk_file


def window_days(out: Path) -> int:
    """Estimation window (sessions) that a run resolved, read from its report.txt."""
    e = engine_of(out)
    m = re.search(e.window_pattern, (out / "report.txt").read_text(encoding="utf-8", errors="replace"))
    if not m:
        raise ValueError(f"no estimation window in {rel(out / 'report.txt')}: update window_pattern of "
                         f"ENGINES['{e.key}'] in atam/source/engines.py")
    return int(m.group(1))


def forecasts(out: Path) -> dict[str, dict[str, str]]:
    """The run's forecast file in the shared schema, keyed by forecast_time."""
    _, rows = read_csv(risk_path(out))
    return {r["forecast_time"]: r for r in rows}
