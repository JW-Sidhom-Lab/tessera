"""Plot CNA segment-mean reconstruction figures for the NoLOH variant.

Reads `*_cna_metrics.pkl` files written by `get_cna_loss_metrics.py` and
produces predicted vs actual segment-mean scatter plots, one panel per
attention-block configuration. The NoLOH variant has no LOH head, so this
script only emits the segment-mean scatter (no LOH ROC).

Inputs
------
``cna_loss/*_cna_metrics.pkl`` (one per trained NoLOH model)

Output
------
``plots/cna_analysis/cna_predicted_vs_actual.png``

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

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
LOSS_FILES_DIR = os.environ.get("LOSS_DIR", "cna_loss/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "plots/cna_analysis")
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

COLOR_PALETTE = {
    "attn_0": "#777777",
    "attn_1": "#88CCEE",
    "attn_2": "#DDAA33",
    "attn_3": "#AA3377",
}

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
# Predicted vs actual segment mean
# ============================================================================
n_models = len(model_order)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]

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

plt.suptitle("CNA Segment Mean (NoLOH)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
out_path = os.path.join(OUTPUT_DIR, "cna_predicted_vs_actual.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out_path}")
