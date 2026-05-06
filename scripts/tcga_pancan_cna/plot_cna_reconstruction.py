"""Plot CNA segment-mean reconstruction and LOH classification figures.

Reads `*_cna_metrics.pkl` files written by `get_cna_loss_metrics.py` and
produces:

  - ``cna_predicted_vs_actual.png`` -- predicted vs actual segment mean
    scatter plots, one panel per attention-block configuration. Used in
    Figure 2 a-b.
  - ``cna_loh_roc_curves.png`` -- LOH classification ROC curves (only
    when the LOH head is present in the metrics file). Used in Figure 2 c.

Inputs
------
``cna_loss/*_cna_metrics.pkl`` (one per trained model)

Outputs
-------
``plots/cna_analysis/cna_predicted_vs_actual.png``
``plots/cna_analysis/cna_loh_roc_curves.png``

Usage
-----
    python plot_cna_reconstruction.py                        # defaults
    LOSS_DIR=cna_loss/ OUTPUT_DIR=plots/cna_analysis python plot_cna_reconstruction.py
    ATTN_BLOCKS_TO_INCLUDE=0,1,2 python plot_cna_reconstruction.py
"""

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import auc, roc_curve

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
LOSS_FILES_DIR = os.environ.get("LOSS_DIR", "cna_loss/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "plots/cna_analysis")
# Comma-separated attention-block counts to include; empty = include all.
ATTN_BLOCKS_TO_INCLUDE = (
    [int(x) for x in os.environ["ATTN_BLOCKS_TO_INCLUDE"].split(",")]
    if os.environ.get("ATTN_BLOCKS_TO_INCLUDE")
    else [0, 1, 2]
)
MATPLOTLIB_BACKEND = os.environ.get("MATPLOTLIB_BACKEND")

if MATPLOTLIB_BACKEND:
    matplotlib.use(MATPLOTLIB_BACKEND, force=True)

np.random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per-attention-block colour palette.
COLOR_PALETTE = {
    "attn_0": "#777777",
    "attn_1": "#88CCEE",
    "attn_2": "#DDAA33",
    "attn_3": "#AA3377",
}

# Subsample cap for the scatter plot (full data still drives the metric).
SCATTER_MAX_POINTS = 20_000

# ============================================================================
# Load all metric files
# ============================================================================
files = sorted(f for f in os.listdir(LOSS_FILES_DIR) if f.endswith(".pkl"))
if not files:
    raise SystemExit(f"No pickle files found in {LOSS_FILES_DIR}")

all_models_data = {}
for pkl_file in files:
    with open(os.path.join(LOSS_FILES_DIR, pkl_file), "rb") as f:
        data = pickle.load(f)
    label = f"attn_{data['cna_attention_blocks']}"
    all_models_data[label] = data
    print(f"Loaded {label}")

model_order = sorted(all_models_data, key=lambda x: all_models_data[x]["cna_attention_blocks"])
if ATTN_BLOCKS_TO_INCLUDE:
    model_order = [m for m in model_order
                   if all_models_data[m]["cna_attention_blocks"] in ATTN_BLOCKS_TO_INCLUDE]
print(f"Plotting models: {model_order}")

# ============================================================================
# Figure 1: Predicted vs actual segment mean
# ============================================================================
n_models = len(model_order)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]

# Shared axis limits across panels for direct visual comparison.
mins, maxs = [], []
for model in model_order:
    d = all_models_data[model]
    mins.extend([float(d["valid_actual"].min()), float(d["valid_predictions"].min())])
    maxs.extend([float(d["valid_actual"].max()), float(d["valid_predictions"].max())])
global_min, global_max = min(mins), max(maxs)
print(f"Shared axis range: [{global_min:.3f}, {global_max:.3f}]")

for idx, model in enumerate(model_order):
    data = all_models_data[model]
    valid_preds = data["valid_predictions"]
    valid_actual = data["valid_actual"]
    pearson_r, _ = stats.pearsonr(valid_actual, valid_preds)

    if len(valid_actual) > SCATTER_MAX_POINTS:
        sub = np.random.choice(len(valid_actual), SCATTER_MAX_POINTS, replace=False)
        plot_actual, plot_preds = valid_actual[sub], valid_preds[sub]
    else:
        plot_actual, plot_preds = valid_actual, valid_preds

    ax = axes[idx]
    ax.scatter(plot_actual, plot_preds, alpha=0.1, s=1, c=COLOR_PALETTE[model])
    ax.plot([global_min, global_max], [global_min, global_max], "r--", linewidth=2)
    ax.set_xlabel("Actual Segment Mean (log2 ratio)", fontsize=11)
    ax.set_ylabel("Predicted Segment Mean (log2 ratio)", fontsize=11)
    ax.set_xlim([global_min, global_max])
    ax.set_ylim([global_min, global_max])
    ax.set_title(f"attn_blocks = {data['cna_attention_blocks']}\nPearson r = {pearson_r:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

plt.suptitle("CNA Segment Mean", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
out_path = os.path.join(OUTPUT_DIR, "cna_predicted_vs_actual.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out_path}")


# ============================================================================
# Figure 2: LOH ROC curves (skipped if any model lacks the LOH head)
# ============================================================================
loh_available = all(all_models_data[m]["valid_loh_predictions"] is not None
                    for m in model_order)
if not loh_available:
    print("LOH not available in all metric files; skipping ROC plot.")
    raise SystemExit(0)

fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]

for idx, model in enumerate(model_order):
    data = all_models_data[model]
    valid_loh_preds = data["valid_loh_predictions"]
    valid_loh_actual = data["valid_loh_actual"]
    ax = axes[idx]

    if len(valid_loh_preds.shape) > 1:
        n_conditions = valid_loh_preds.shape[1]
        colors_roc = plt.cm.tab10(np.linspace(0, 1, n_conditions))
        for cond_idx in range(n_conditions):
            fpr, tpr, _ = roc_curve(valid_loh_actual[:, cond_idx],
                                    valid_loh_preds[:, cond_idx])
            ax.plot(fpr, tpr,
                    label=f"Condition {cond_idx}: AUC={auc(fpr, tpr):.4f}",
                    color=colors_roc[cond_idx], linewidth=2)
    else:
        fpr, tpr, _ = roc_curve(valid_loh_actual, valid_loh_preds)
        ax.plot(fpr, tpr,
                label=f"AUC={auc(fpr, tpr):.4f}",
                color=COLOR_PALETTE[model], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"attn_blocks = {data['cna_attention_blocks']}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.suptitle("CNA LOH Classification", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
out_path = os.path.join(OUTPUT_DIR, "cna_loh_roc_curves.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out_path}")
