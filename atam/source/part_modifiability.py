"""Part 5 - modifiability: scenarios M1 (add a fourth VaR model) and M2 (move to the target time grid).

Both are design-time scenarios, so the evidence is read from the source code instead of measured
at run time. Every judgement is a pattern search configured in CHECKS below and re-run on every
call: when the engines change, the numbers and the listed files follow. If a check stops matching
because code was renamed, update its pattern here (the report prints every file it matched).

  M1  files outside the model layer that name a specific model's result (a fourth model means
      touching each of them); a common model interface would bring this number down to the
      registration point.
  M2  whether the target grid of the project (5-second sampling, forecasts every minute for a
      5-minute horizon) is reachable through configuration; each matched blocker is a hard-wired
      decision that needs a code change.
Code-structure metrics (modules, lines, include graph) are collected for context.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from common import REPO, RESULTS, Measurement, Section, rel, write_csv, write_text
from engines import ENGINES, default_variants

# --------------------------------------------------------------------------- edit here when the code changes
# The model layer of each engine: the files that implement its VaR models. garch_t's portfolio.cpp is
# the GARCH-t/EVT model itself (the baseline is a separate model since the common interface); its
# models.cpp and ewma's models.cpp hold the classes behind riskcommon::PortfolioModel and their
# registration (make_models).
MODEL_LAYER = {"ewma": ("ewma", "risk", "seasonality", "models"),
               "garch_t": ("garch", "risk", "seasonality", "portfolio", "models"),
               "realized_caviar": ("model",)}
HEADER_DIR = {"ewma": "ewmar", "garch_t": "garcht", "realized_caviar": "rcaviar"}
MODEL_FIELDS = {  # code outside the model layer that names one specific model (a fourth model would touch it)
    "ewma": [r"\b(?:EwmaNormalModel|EwmaStudentModel|RiskMetricsModel)\b",  # a model class
             r"\b(?:models|var_models|berkowitz)(?:\(\))?\s*\[\s*\d+\s*\]",   # a model by its position
             r"(?<!\])\.(?:ref|normal|student)\s*\.\s*(?:sigma|m)\b"],          # a named result field (old design)
    "garch_t": [r"\b(?:GarchPortfolioModel|RiskMetricsModel)\b",
                r"\b(?:models|var_models|berkowitz)(?:\(\))?\s*\[\s*\d+\s*\]",
                r"(?<!\])\.(?:garch|ewma)\b(?![_(])", r"std::vector<RiskMeasure>\s+(?:garch|ewma)\b"],
    "realized_caviar": [r"\bm\s*==\s*[0-2]\b", r"\bBaseline(?:State|Calibration)\b",
                        r'"(?:EWMA-N|GARCH-t|Realized-CAViaR)'],  # models selected by position in the replay
}
COMMON_INTERFACE = (r"\bvirtual\b|\bconcept\b|template\s*<\s*(?:class|typename)\s+Model\b"
                    r"|riskcommon::PortfolioModel\b")
M2_BLOCKERS = [  # (pattern, files, meaning, unless): a hard-wired decision, unless the `unless` pattern shows a way
    (r"HH:MM,BAR_MIN,BARS", "source/apps/*.cpp",
     "bar length is set in whole minutes (--session HH:MM,BAR_MIN,BARS): 5-second sampling needs code",
     r"--resample"),
]
M2_ENABLERS = [  # (pattern, files, meaning): evidence in code that the target grid is built in
    (r"timestamp\s*-\s*prices\[i-1\]\.timestamp\s*!=\s*5\b", "source/src/replay.cpp",
     "high-frequency mode (--prices) takes 5-second prices"),
    (r"/\s*60\s*%\s*5\b", "source/src/replay.cpp",
     "forecasts every minute in five 1-minute phases of a 5-minute horizon"),
    (r"--resample", "source/apps/*.cpp",
     "5-second (or any finer) samples are aggregated into the bars of any --session grid (--resample)"),
]
M2_NOTES = [  # hard-wired choices that do not block the target grid but would block another one
    (r"requires 15-minute bars", "source/src/shared_data.cpp", "the shared-data mode accepts 15-minute bars only"),
    (r"next_slot\s*=\s*\(\s*slot(?:\[t\])?\s*\+\s*1\s*\)\s*%\s*K", "source/src/replay.cpp",
     "the forecast horizon is one bar: a 5-minute VaR every minute is five runs on 5-minute grids opening one "
     "minute apart (--session 10:00,5,84 ... 10:04,5,83), each with non-overlapping targets, as CAViaR's phases"),
]
M1_EXCLUDE = ("benchmark",)  # programs that time a model but would not change to add one
SIBLING_SOURCES = r"(?:\.\./\.\./(?:ewma|garch_t)/source|\$\{GARCH_SOURCE\})/src/(\w+)\.cpp"  # CMakeLists reuse
COMMON_LIBRARY = r"add_subdirectory\([^)]*\.\./\.\./common\b"  # the shared library (interface, filter)
# ------------------------------------------------------------------------------------------------------------


def _sources(engine: str) -> list[Path]:
    root = REPO / ENGINES[engine].folder / "source"
    return sorted(p for p in root.rglob("*") if p.suffix in (".cpp", ".hpp") and "build" not in p.parts)


def _module(p: Path) -> str:
    return p.stem


def code_metrics(engine: str) -> dict:
    files = _sources(engine)
    loc = {"include": 0, "src": 0, "apps": 0, "tests": 0}
    edges = set()
    hdr = HEADER_DIR[engine]
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        part = next((k for k in loc if k in p.parts), None)
        if part:
            loc[part] += sum(1 for line in text.splitlines() if line.strip())
        if "tests" in p.parts:
            continue
        for m in re.finditer(rf'#include\s+"{hdr}/(\w+)\.hpp"', text):
            if m.group(1) != _module(p) and m.group(1) != hdr:
                edges.add((_module(p), m.group(1)))
    modules = sorted({_module(p) for p in files if "src" in p.parts})
    return {"modules": modules, "loc": loc, "edges": sorted(edges)}


def _code(text: str) -> str:
    """C++ source without its comments (string literals are kept: they can name a model too)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def m1_files(engine: str) -> tuple[list[str], bool]:
    layer = set(MODEL_LAYER[engine])
    pats = [re.compile(x) for x in MODEL_FIELDS[engine]]
    hits, interface = [], False
    for p in _sources(engine):
        if "tests" in p.parts:
            continue
        text = _code(p.read_text(encoding="utf-8", errors="replace"))
        interface |= bool(re.search(COMMON_INTERFACE, text))
        if _module(p) in layer or _module(p) in M1_EXCLUDE:
            continue
        if any(x.search(text) for x in pats):
            hits.append(rel(p))
    return hits, interface


def m2_blockers(engine: str) -> list[tuple[str, str]]:
    root = REPO / ENGINES[engine].folder
    found = []
    for pattern, glob, meaning, unless in M2_BLOCKERS:
        files = sorted(root.glob(glob))
        if unless and any(re.search(unless, p.read_text(encoding="utf-8", errors="replace")) for p in files):
            continue  # the code offers a way around it
        for p in files:
            text = p.read_text(encoding="utf-8", errors="replace")
            m = re.search(pattern, text)
            if m:
                line = text[: m.start()].count("\n") + 1
                found.append((f"{rel(p)}:{line}", meaning))
                break
    return found


def m2_enablers(engine: str, checks: list | None = None) -> list[tuple[str, str]]:
    root = REPO / ENGINES[engine].folder
    found = []
    for pattern, glob, meaning in (M2_ENABLERS if checks is None else checks):
        for p in sorted(root.glob(glob)):
            text = p.read_text(encoding="utf-8", errors="replace")
            m = re.search(pattern, text)
            if m:
                found.append((f"{rel(p)}:{text[: m.start()].count(chr(10)) + 1}", meaning))
                break
    return found


def sibling_reuse(engine: str) -> list[str]:
    """Source files of other modules compiled into this one (coupling at source level)."""
    cm = REPO / ENGINES[engine].folder / "source" / "CMakeLists.txt"
    if not cm.is_file():
        return []
    text = cm.read_text(encoding="utf-8", errors="replace")
    return sorted({m.group(0).replace("${GARCH_SOURCE}", "garch_t/source").replace("../../", "")
                   for m in re.finditer(SIBLING_SOURCES, text)})


def run(fresh: bool = False) -> list[Measurement]:
    ms: list[Measurement] = []
    s = Section("M1 / M2 - modifiability (static analysis of the engines' source code)")
    rows_metrics, rows_m1, rows_m2 = [], [], []
    for v in default_variants():
        ek = v.engine
        e = ENGINES[ek]
        cm = code_metrics(ek)
        hits, interface = m1_files(ek)
        blockers = m2_blockers(ek)
        enablers = m2_enablers(ek)
        notes = m2_enablers(ek, M2_NOTES)
        reuse = sibling_reuse(ek)
        # yes: no blocker and the grid found in code; no: a blocker; unknown (nan): neither, the patterns are stale
        config_only = 0 if blockers else 1 if enablers else math.nan
        rows_metrics.append({"engine": ek, "modules": len(cm["modules"]), "include_edges": len(cm["edges"]),
                             **{f"loc_{k}": n for k, n in cm["loc"].items()},
                             "common_model_interface": interface, "module_list": " ".join(cm["modules"])})
        rows_m1 += [{"engine": ek, "file": f} for f in hits]
        rows_m2 += [{"engine": ek, "where": w, "blocker": m} for w, m in blockers]
        s.add(f"{e.label}: {len(cm['modules'])} modules, {len(cm['edges'])} include edges, "
              f"{cm['loc']['src'] + cm['loc']['include']} lines of library code, {cm['loc']['apps']} of programs, "
              f"{cm['loc']['tests']} of tests")
        s.add(f"  M1: common model interface: {'yes' if interface else 'no'}; files outside the model layer "
              f"({', '.join(MODEL_LAYER[ek])}) that name a specific model's result: {len(hits)}")
        for f in hits:
            s.add(f"      {f}")
        s.add(f"  M2: target grid reachable by configuration: "
              f"{'no' if config_only == 0 else 'yes' if config_only == 1 else 'unknown (no pattern matched)'}")
        for w, m in blockers + enablers + notes:
            s.add(f"      {w}: {m}")
        cmake = REPO / e.folder / "source" / "CMakeLists.txt"
        shared = cmake.is_file() and bool(re.search(COMMON_LIBRARY, cmake.read_text(encoding="utf-8", errors="replace")))
        s.add(f"  shared library common/ (model interface, RiskMetrics baseline, jump filter): "
              f"{'built and linked' if shared else 'not used'}")
        if reuse:
            s.add(f"  reuse: compiles {len(reuse)} source files of other modules: {', '.join(reuse)}")
        ms += [Measurement("M1", v.key, "m1_files_outside_model", len(hits), "files",
                           ("no common model interface; " if not interface else "")
                           + f"files naming a specific model: {', '.join(Path(h).name for h in hits)}"),
               Measurement("M1", v.key, "common_model_interface", 1 if interface else 0, "yes/no"),
               Measurement("M2", v.key, "m2_config_only", config_only, "yes/no",
                           "; ".join(f"{m} [{w}]" for w, m in blockers + enablers + notes)
                           or "no pattern matched: update M2_BLOCKERS / M2_ENABLERS"),
               Measurement("M1", v.key, "reused_sibling_sources", len(reuse), "files",
                           ", ".join(reuse) or "builds from its own sources only"),
               Measurement("M1", v.key, "uses_common_library", 1 if shared else 0, "yes/no")]
    write_csv(RESULTS / "modifiability" / "code_metrics.csv", list(rows_metrics[0].keys()), rows_metrics)
    write_csv(RESULTS / "modifiability" / "m1_model_specific_files.csv", ["engine", "file"], rows_m1)
    write_csv(RESULTS / "modifiability" / "m2_blockers.csv", ["engine", "where", "blocker"], rows_m2)
    write_text(RESULTS / "modifiability" / "report.txt", s.text())
    print("   " + "\n   ".join(s.text().splitlines()))
    return ms
