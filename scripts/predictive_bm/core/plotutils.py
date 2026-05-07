"""Shared style constants and formatting helpers for the plot library."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- color palette --------------------------------------------------------
# Treatment arms
ARM1_COLOR = "#1f77b4"   # blue:  arm=1 (FOLFOX, FOLFIRINOX)
ARM0_COLOR = "#ff7f0e"   # orange: arm=0 (FOLFIRI, Gem/Abraxane)

# Tau tertiles  (Q1 = top τ̂ = arm-1-favored, Q3 = bottom = arm-0-favored)
TERT_COLORS = ["#2ca02c", "#1f77b4", "#d62728"]
TERT_LABELS = ["Q1 (top τ̂)", "Q2 (mid τ̂)", "Q3 (bot τ̂)"]

# Threshold split
ABOVE_COLOR = "#1f77b4"   # blue:  τ̂ > τ̂_0  → arm 1 favored
BELOW_COLOR = "#d62728"   # red:   τ̂ < τ̂_0  → arm 0 favored

# Recommendation × actual
CONCORDANT_COLOR = "#2ca02c"   # green
DISCORDANT_COLOR = "#d62728"   # red


# ---- typography / matplotlib defaults -------------------------------------
def apply_default_style():
    mpl.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.titlesize":  12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ---- formatting helpers ---------------------------------------------------
def format_p(p: float) -> str:
    """0.0123 → '0.012'; 4.83e-5 → '4.8×10⁻⁵'; 1e-12 → '1×10⁻¹²'."""
    if p is None or (isinstance(p, float) and (p != p)):
        return "NA"
    if p < 1e-3:
        m, e = f"{p:.0e}".split("e")
        return f"{float(m):g}×10⁻{abs(int(e))}"
    return f"{p:.3f}"


def format_hr(hr: float, lo: float, hi: float) -> str:
    """0.685, 0.586, 0.814 → '0.69 [0.59, 0.81]'."""
    return f"{hr:.2f} [{lo:.2f}, {hi:.2f}]"


def save_fig(fig, out_base, dpi: int = 200):
    """Save a figure to PDF + PNG at the same path stem."""
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
