"""Figure style of the ATAM evidence: one look for every chart (light surface, 16:9, for slides and print).

Colors follow the engine, never its rank: blue = EWMA, orange = GARCH-t, aqua = Realized-CAViaR
(validated categorical slots 1-3). A non-default variant keeps its engine's hue and is drawn
dashed, so identity never depends on color alone; every multi-series chart also has a legend.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from common import FIGURES, rel  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ENGINE_COLOR = {"ewma": "#2a78d6", "garch_t": "#eb6834", "realized_caviar": "#1baf7a"}
STATUS_COLOR = {"MET": "#0ca30c", "NOT MET": "#d03b3b", "PARTIAL": "#fab219", "PENDING": "#e1e0d9",
                "N/A": "#e1e0d9", "REPORTED": "#b7d3f6"}
STATUS_MARK = {"MET": "✓", "NOT MET": "✗", "PARTIAL": "~", "PENDING": "…", "N/A": "–",
               "REPORTED": "i"}

_FONT = next((f for f in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans")
              if any(f == x.name for x in font_manager.fontManager.ttflist)), "DejaVu Sans")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_FONT, "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 11,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 100,
    "text.color": INK,
    "svg.hashsalt": "atam",  # deterministic SVG ids: a rerun with the same numbers changes no file
})


SLIDE = (12.0, 6.75)  # inches, 16:9


def variant_style(variant) -> dict:
    """Line style of an engine variant: engine hue; dashed when it is not the default configuration."""
    return {"color": ENGINE_COLOR[variant.engine], "linestyle": "-" if variant.default else (0, (4, 2))}


def heading(fig, title: str, subtitle: str = "") -> None:
    """Figure title and one-line takeaway, left-aligned above the plots."""
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=11.5, color=INK_2)


def save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{name}.png"
    fig.savefig(png, dpi=200, metadata={"Software": None})
    fig.savefig(FIGURES / f"{name}.svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return rel(png)
