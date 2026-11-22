# ATAM evaluation of the intraday risk engines

Architecture evaluation of the three engines of this repository with the **Architecture Tradeoff
Analysis Method** (ATAM) of the Software Engineering Institute, Carnegie Mellon University. Each
engine is one architecture:

| Engine | Folder | Architecture in one line |
|---|---|---|
| EWMA | `ewma/` | portfolio engine in layers, models behind the common interface, a data-quality layer (NORMAL / DEGRADED / FREEZE) |
| GARCH-t | `garch_t/` | portfolio engine: GARCH per asset, dynamic correlation, EVT tail, options by Monte Carlo off the bar thread, models behind the common interface |
| Realized-CAViaR | `realized_caviar/` | single-instrument engine; CAViaR plus matched EWMA-N, EWMA-t and GARCH-t baselines on the same targets |

The three compile the shared library `common/` (the model interface, the RiskMetrics baseline and the
jump filter that each of them runs by default).

The evaluation follows the four ATAM steps asked for the project:

1. **Business drivers**: below.
2. **Quality ruler and risks**: the utility tree, [`utility_tree.csv`](utility_tree.csv), with a
   measurable response and an (importance, difficulty) priority per scenario.
3. **Architectural scenarios**: each row of the utility tree in six parts (source, stimulus,
   artifact, environment, response, response measure).
4. **Evaluation**: `source/` runs the engines, measures every scenario, and writes the scorecard,
   the risks, non-risks, sensitivity points and trade-off points to `results/`.

How the nine steps of the method (Kazman, Klein and Clements, 2000) map to this folder:

| ATAM step | Where |
|---|---|
| 1-2. Present the method and the business drivers | this README |
| 3. Present the architectures | `architecture/` (diagrams of the three engines) and the table above |
| 4. Identify the architectural approaches | `ENGINES` and `VARIANTS` in `source/engines.py`: each variant switches one decision |
| 5. Generate the utility tree | `utility_tree.csv` (priorities set by the team) |
| 6. Analyse the approaches | the parts in `source/`; findings in `results/report.txt` |
| 7. Brainstorm and prioritize scenarios with the stakeholders | open: the utility tree and its thresholds are to be validated with the advisor: are the scenarios, priorities and thresholds right, and which data policy should the statistical comparison use? |
| 8. Analyse again with the new priorities | rerun `run.bat` after editing `utility_tree.csv`: thresholds are data |
| 9. Present the results | `results/report.txt`, `results/figures/`, `results/latex/` |

Every number in `results/` is produced by `run.bat` / `run.sh`; nothing is typed by hand.

## Business drivers

- **Purpose.** Forecast intraday VaR 99% during the session and answer the research question:
  does Realized-CAViaR forecast better than EWMA-Normal and GARCH-t, at an acceptable computational
  cost?
- **Stakeholders.** Risk desk (fresh, trustworthy VaR), researchers (a fair, reproducible
  comparison), advisor and examiners (sound engineering evidence).
- **Constraints.** C++ engines, single process, offline calibration, no message broker or database
  (`DECISIONS.md`); shared 15-minute data (`data/`); a three-person team and one semester.
- **Quality attributes, in priority order.** Comparability and temporal correctness of the
  forecasts, robustness to a degraded feed, reproducibility, performance, modifiability.

## Run

```bat
atam\run.bat                               :: every part (Windows); engine runs are cached
atam\run.bat --fresh                       :: run every engine again
atam\run.bat --only robustness,scorecard   :: some parts only
```

`run.sh` does the same on Linux/macOS. Requirements: Python 3.10+ with `matplotlib` and `numpy`
(`requirements.txt`, installed by the scripts when missing), CMake and a C++20 compiler. EWMA and
GARCH-t use their versioned programs in `bin/`; Realized-CAViaR is built from source
(`realized_caviar/build.bat` the first time, an incremental rebuild afterwards, because it compiles
sources of ewma and garch_t). A full run takes about 45 minutes on a desktop i7 (`results/environment.txt`), most of it the
fault catalogue (every engine on 30 faulty copies, Realized-CAViaR once per stock); a cached rerun
takes a few seconds. The engine runs are kept in `atam/work/`; set `ATAM_WORK` to another folder to
keep them elsewhere (for example when an antivirus stops freshly built programs from writing in the
user's documents).

## What it produces

```
results/
├── report.txt          start here: scorecard, findings, then each part's report
├── scorecard.csv       scenario x engine: value, threshold, status, detail
├── findings.csv        risks (R-), sensitivity points (S-), trade-offs (T-), non-risks (N)
├── environment.txt     machine, commit and SHA-256 of every engine program used
├── figures/            16:9 figures for slides and the monograph (PNG and SVG)
├── latex/              utility tree, scorecard and findings as \input-ready LaTeX (abnTeX2 quadro)
└── <part>/             evidence of each part: CSV tables and report.txt
```

| Part | Scenarios | How it is measured |
|---|---|---|
| `correctness` | C1, Rp1 | prices after a cut time are replaced (perturbation) or removed (truncation): nothing issued before the cut may change; three identical runs compared cell by cell |
| `comparability` | Cp1 | the EWMA(0.94) baseline that EWMA and GARCH-t both compute, forecast by forecast, on synthetic and B3 data; the shared targets of Realized-CAViaR's three models |
| `robustness` | R1, R2 | the faulty feed against the clean one (for Realized-CAViaR also the days its daily calibration lands on another optimum); the VaR at scripted genuine shocks with and without each engine's protection (Realized-CAViaR also: whether the shock reaches its features) |
| `faults` | R3 | faulty copies of both datasets written by `data/tools/inject_faults.py`, one kind of fault per copy plus all of them together, each engine against its own clean run; the faults flagged by the EWMA data-quality layer against the list of every fault injected |
| `performance` | P1-P4 | per-bar latency from the replays, the benchmark programs, refit times, model-state size |
| `modifiability` | M1, M2 | static analysis of the sources: files touched to add a model, hard-wired time-grid decisions, reuse |
| `scorecard` | all | joins the measurements with the utility tree; writes findings, figures and LaTeX |

**Statuses in the scorecard.** MET / NOT MET against the threshold of the utility tree; PARTIAL
when the threshold falls inside the range of the repeated runs (widened by 25%), so another run
could flip the verdict; REPORTED for a measure without a threshold; N/A when the architecture has no
such capability or does not measure it the way the scenario asks.

**Measurement caveats, stated in the report.**

- Timings depend on the machine and its state; on the same machine they vary by up to a factor
  of two between sessions. Use the same quiet machine, on mains power, for numbers that will be quoted.
- The portfolio engines time a whole bar cycle of 15 stocks; Realized-CAViaR times the model stage of
  one instrument in batches of 1000 (percentiles of batch means). Only the matched model-stage
  benchmark compares the three models under one boundary.
- R1 counts single-stock forecasts for Realized-CAViaR and portfolio forecasts for the others; a
  forecast issued from only one of the two feeds counts as off.

## When the engines change

The evaluation is meant to be rerun after every change to the engines, with few or no edits:

- **Rebuild, then rerun.** A run is cached by command line, SHA-256 of the program, SHA-256 of every
  input file and the machine, so a rebuilt engine is replayed automatically and timings from another
  machine are never reused. `--fresh` forces every run again.
- **Interfaces live in one place.** Everything assumed about an engine (program, folders, output
  files, VaR column, calibration files, single-instrument mode, how the window is printed) is in
  `ENGINES` in [`source/engines.py`](source/engines.py). If an engine renames a column or a file, the
  run stops with the name of the missing item and the entry to edit.
- **New output format.** An engine whose output is not in the shared schema gets an adapter
  (`ADAPTERS` in `engines.py`; `caviar_adapter` is the example).
- **Benchmark text.** `BENCH_PARSERS` in `source/part_performance.py` parses each benchmark program.
- **Static checks.** The patterns of M1/M2 are at the top of `source/part_modifiability.py`; the
  report lists every file each pattern matched, so a stale pattern is visible.
- **Findings follow the evidence.** Each finding in `source/part_scorecard.py` is a rule over the
  measurements: a problem that is fixed disappears from the next report without editing the rule.
- **Thresholds are data.** Priorities and thresholds are in `utility_tree.csv`.
- **New scenario.** Add a row to `utility_tree.csv` and a `Measurement` with the same `metric` in
  the part that measures it.

## References

- R. Kazman, M. Klein, P. Clements. *ATAM: Method for Architecture Evaluation*. CMU/SEI-2000-TR-004,
  Software Engineering Institute, Carnegie Mellon University, 2000.
- P. Clements, R. Kazman, M. Klein. *Evaluating Software Architectures: Methods and Case Studies*.
  Addison-Wesley, 2002.
- L. Bass, P. Clements, R. Kazman. *Software Architecture in Practice*, 4th ed. Addison-Wesley, 2021.
