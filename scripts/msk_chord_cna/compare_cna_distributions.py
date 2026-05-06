"""Compare TCGA and MSK-CHORD Segment_Mean distributions across normalisation modes.

Diagnostic figure for the cross-platform CNA-validation step in
Figure 2 d. Shows whether the WES-vs-panel distribution shift is
captured by a global mean/std rescale, or whether the shapes differ
enough (heavy tails, bimodality) that quantile normalisation is needed.
The manuscript ultimately uses quantile normalisation (see
``get_cna_loss_acc.py``); this script is the evidence behind that
choice.

Inputs
------
``../data/tcga/{train,valid}_data_cna.csv``
``../data/msk_chord/cna.csv`` and ``cna_panel_filtered.csv``

Outputs
-------
``plots/cna_distribution_comparison.png``
``plots/cna_distribution_summary.csv``

Usage
-----
    python compare_cna_distributions.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
TCGA_PATHS = [
    os.environ.get("TCGA_TRAIN_CNA", "../data/tcga/train_data_cna.csv"),
    os.environ.get("TCGA_VALID_CNA", "../data/tcga/valid_data_cna.csv"),
]
MSK_RAW_PATH = os.environ.get("MSK_RAW", "../data/msk_chord/cna.csv")
MSK_PANEL_PATH = os.environ.get("MSK_PANEL", "../data/msk_chord/cna_panel_filtered.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "plots")
MATPLOTLIB_BACKEND = os.environ.get("MATPLOTLIB_BACKEND")

if MATPLOTLIB_BACKEND:
    matplotlib.use(MATPLOTLIB_BACKEND, force=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load(paths, label):
    if isinstance(paths, str):
        paths = [paths]
    vals = []
    for p in paths:
        df = pd.read_csv(p, usecols=["Segment_Mean"])
        vals.append(df["Segment_Mean"].astype(float).values)
    arr = np.concatenate(vals)
    print(f"  {label}: {arr.size:,} segments loaded from {paths}")
    return arr


def summarize(arr, label):
    pct = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return {
        "label": label, "n": arr.size,
        "mean": float(arr.mean()), "std": float(arr.std()),
        "median": float(np.median(arr)),
        "IQR": float(pct[4] - pct[2]),
        "p1": float(pct[0]), "p5": float(pct[1]), "p25": float(pct[2]),
        "p50": float(pct[3]),
        "p75": float(pct[4]), "p95": float(pct[5]), "p99": float(pct[6]),
        "min": float(arr.min()), "max": float(arr.max()),
        "skew": float(spstats.skew(arr)),
        "kurtosis": float(spstats.kurtosis(arr)),
        "frac_abs_gt_0.5": float(np.mean(np.abs(arr) > 0.5)),
        "frac_abs_gt_1.0": float(np.mean(np.abs(arr) > 1.0)),
    }


def linear_rescale(x, target_mean, target_std):
    return (x - x.mean()) / x.std() * target_std + target_mean


def quantile_normalize_to_tcga(vals, tcga_sorted):
    n = len(vals)
    ranks = spstats.rankdata(vals, method="average")
    q = (ranks - 0.5) / n
    tcga_q = np.linspace(0.0, 1.0, len(tcga_sorted))
    return np.interp(q, tcga_q, tcga_sorted)


def ks_summary(a, b, label):
    D, _ = spstats.ks_2samp(a, b, mode="asymp")
    return f"  KS {label}: D={D:.4f}"


def ecdf(x):
    s = np.sort(x)
    return s, np.arange(1, len(s) + 1) / len(s)


# ============================================================================
# Load + summarise
# ============================================================================
print("Loading CNA Segment_Mean values...")
tcga = _load(TCGA_PATHS, "TCGA")
msk_raw = _load(MSK_RAW_PATH, "MSK raw")
msk_pnl = _load(MSK_PANEL_PATH, "MSK panel")

rows = [summarize(a, lbl) for a, lbl in
        [(tcga, "TCGA"), (msk_raw, "MSK_raw"), (msk_pnl, "MSK_panel")]]
df = pd.DataFrame(rows).set_index("label").T
print("\nSummary statistics (Segment_Mean):")
print(df.round(4).to_string())
df.to_csv(os.path.join(OUTPUT_DIR, "cna_distribution_summary.csv"))

# ============================================================================
# Apply normalisation modes
# ============================================================================
msk_raw_rs = linear_rescale(msk_raw, tcga.mean(), tcga.std())
msk_pnl_rs = linear_rescale(msk_pnl, tcga.mean(), tcga.std())

print("\nSorting TCGA Segment_Mean for quantile normalisation...")
tcga_sorted = np.sort(tcga)
msk_raw_qn = quantile_normalize_to_tcga(msk_raw, tcga_sorted)
msk_pnl_qn = quantile_normalize_to_tcga(msk_pnl, tcga_sorted)

print("\nShape-distance (Kolmogorov-Smirnov) vs TCGA:")
print(ks_summary(msk_raw, tcga, "MSK raw   (none)    "))
print(ks_summary(msk_raw_rs, tcga, "MSK raw   (linear)  "))
print(ks_summary(msk_raw_qn, tcga, "MSK raw   (quantile)"))
print(ks_summary(msk_pnl, tcga, "MSK panel (none)    "))
print(ks_summary(msk_pnl_rs, tcga, "MSK panel (linear)  "))
print(ks_summary(msk_pnl_qn, tcga, "MSK panel (quantile)"))

# ============================================================================
# Plot: 3 (modes) x 3 (panels) figure
# ============================================================================
clip_lim = 3.0
bins = np.linspace(-clip_lim, clip_lim, 120)
col_tcga, col_mskr, col_mskp = "#117733", "#CC3311", "#4477AA"

fig, axes = plt.subplots(3, 3, figsize=(17, 13))
modes = [
    ("None (raw)", msk_raw, msk_pnl),
    ("Linear rescale", msk_raw_rs, msk_pnl_rs),
    ("Quantile norm", msk_raw_qn, msk_pnl_qn),
]

# Row 1: histograms
for ax, (title, mr, mp) in zip(axes[0], modes):
    ax.hist(np.clip(tcga, -clip_lim, clip_lim), bins=bins, alpha=0.5,
            density=True, color=col_tcga, label=f"TCGA (n={tcga.size:,})")
    ax.hist(np.clip(mr, -clip_lim, clip_lim), bins=bins, alpha=0.5,
            density=True, color=col_mskr, label=f"MSK raw (n={msk_raw.size:,})")
    ax.hist(np.clip(mp, -clip_lim, clip_lim), bins=bins, alpha=0.5,
            density=True, color=col_mskp, label=f"MSK pnl (n={msk_pnl.size:,})")
    ax.set_title(f"{title}\nhistograms")
    ax.set_xlabel("Segment_Mean (clipped to +/-3)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

# Row 2: Q-Q vs TCGA
q = np.linspace(0.001, 0.999, 300)
tq = np.quantile(tcga, q)
for ax, (title, mr, mp) in zip(axes[1], modes):
    ax.plot(tq, np.quantile(mr, q), "o", markersize=3, color=col_mskr,
            alpha=0.7, label="MSK raw")
    ax.plot(tq, np.quantile(mp, q), "s", markersize=3, color=col_mskp,
            alpha=0.7, label="MSK pnl")
    all_lo = min(tq.min(), np.quantile(mr, q).min(), np.quantile(mp, q).min())
    all_hi = max(tq.max(), np.quantile(mr, q).max(), np.quantile(mp, q).max())
    ax.plot([all_lo, all_hi], [all_lo, all_hi], "--", color="k",
            linewidth=0.8, label="y=x")
    ax.set_xlabel("TCGA quantile")
    ax.set_ylabel("MSK quantile")
    ax.set_title(f"{title}\nQ-Q vs TCGA")
    ax.legend(fontsize=8)

# Row 3: ECDFs
for ax, (title, mr, mp) in zip(axes[2], modes):
    for arr, lbl, c in [(tcga, "TCGA", col_tcga),
                        (mr, "MSK raw", col_mskr),
                        (mp, "MSK pnl", col_mskp)]:
        xs, ys = ecdf(np.clip(arr, -clip_lim, clip_lim))
        ax.plot(xs, ys, label=lbl, color=c, linewidth=1.2)
    ax.set_xlabel("Segment_Mean (clipped to +/-3)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"{title}\nECDFs")
    ax.legend(fontsize=8)

fig.suptitle("CNA Segment_Mean: TCGA vs MSK-CHORD, three normalisation modes",
             y=1.00, fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "cna_distribution_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_path}")
print(f"Saved {os.path.join(OUTPUT_DIR, 'cna_distribution_summary.csv')}")
