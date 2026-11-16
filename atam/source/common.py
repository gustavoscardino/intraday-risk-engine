"""Shared helpers of the ATAM evaluation: paths, CSV input/output, statistics and measurements."""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

ATAM_DIR = Path(__file__).resolve().parent.parent  # atam/
REPO = ATAM_DIR.parent
DATA = REPO / "data"
SYNTHETIC = DATA / "synthetic"
B3_REAL = DATA / "b3_real"
DEGRADED_PRICES = SYNTHETIC / "prices_15m_degraded.csv"
WORK = ATAM_DIR / "work"  # raw engine runs, not versioned
RESULTS = ATAM_DIR / "results"  # versioned evidence
FIGURES = RESULTS / "figures"
UTILITY_TREE = ATAM_DIR / "utility_tree.csv"


def rel(path: Path) -> str:
    """Path relative to the repository, with forward slashes (as in the engine reports)."""
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ----------------------------------------------------------------------------- CSV


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.10g}"
    return str(v)


def write_csv(path: Path, header: Sequence[str], rows: Iterable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            if isinstance(row, dict):
                w.writerow([_cell(row.get(h)) for h in header])
            else:
                w.writerow([_cell(v) for v in row])


def to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


# ----------------------------------------------------------------------------- statistics


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default), p in [0, 100]."""
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return math.nan
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- measurements


@dataclass
class Measurement:
    """One measured value of a utility-tree scenario for one engine variant."""

    scenario: str
    variant: str
    metric: str
    value: float
    unit: str = ""
    detail: str = ""


MEASUREMENT_HEADER = ["scenario", "variant", "metric", "value", "unit", "detail"]


def write_measurements(part: str, ms: list[Measurement]) -> Path:
    path = RESULTS / part / "measurements.csv"
    write_csv(path, MEASUREMENT_HEADER, ([m.scenario, m.variant, m.metric, m.value, m.unit, m.detail] for m in ms))
    return path


def read_all_measurements() -> list[Measurement]:
    out: list[Measurement] = []
    for path in sorted(RESULTS.glob("*/measurements.csv")):
        _, rows = read_csv(path)
        for r in rows:
            out.append(Measurement(r["scenario"], r["variant"], r["metric"], to_float(r["value"]), r["unit"], r["detail"]))
    return out


# ----------------------------------------------------------------------------- report text


@dataclass
class Section:
    """A block of the text report, in the layout of the engines' report.txt files."""

    title: str
    lines: list[str] = field(default_factory=list)

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join([self.title, *("  " + l if l else "" for l in self.lines)])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def table(header: Sequence[str], rows: Sequence[Sequence], align: str | None = None) -> list[str]:
    """Fixed-width text table; align is a string of 'l'/'r' per column (default: first left, rest right)."""
    cells = [[str(h) for h in header]] + [[_cell(c) if not isinstance(c, str) else c for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    align = align or ("l" + "r" * (len(header) - 1))
    out = []
    for r in cells:
        out.append("  ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(r, widths, align)).rstrip())
    return out
