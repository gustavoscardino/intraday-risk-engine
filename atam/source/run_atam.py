"""ATAM evaluation of the intraday risk engines.

usage: python run_atam.py [--only PART[,PART...]] [--fresh]

Parts (in order): environment, correctness, comparability, robustness, performance,
modifiability, scorecard. Each part runs the engines it needs (cached in atam/work/), writes its
evidence to atam/results/<part>/ and its figures to atam/results/figures/. The scorecard part
joins the measurements with the utility tree (atam/utility_tree.csv) and writes
atam/results/report.txt.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_measurements  # noqa: E402

PARTS = ["environment", "correctness", "comparability", "robustness", "performance", "modifiability", "scorecard"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="comma-separated parts to run (default: all)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached engine runs and run them again")
    a = ap.parse_args(argv)
    parts = PARTS if not a.only else [p.strip() for p in a.only.split(",") if p.strip()]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        ap.error(f"unknown part(s): {', '.join(unknown)}; choose from {', '.join(PARTS)}")
    for part in parts:
        t0 = time.perf_counter()
        print(f"== {part}")
        module = importlib.import_module(f"part_{part}")
        ms = module.run(fresh=a.fresh)
        if ms is not None:
            write_measurements(part, ms)
        print(f"   done in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
