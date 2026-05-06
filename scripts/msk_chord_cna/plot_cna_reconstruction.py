"""Plot Figure 2 d: predicted vs actual CNA segment mean on MSK-CHORD panel-sequenced data.

Reads the per-model pkl files written by ``get_cna_loss_acc.py`` and
produces a one-row scatter-plot panel per attention-block configuration,
with each panel labeled by its Pearson correlation between predicted
and actual Segment_Mean.

Inputs
------
``cna_loss_panel_filtered/<MODEL>_msk_chord_loss_cna.pkl``

Output
------
``plots/cna_analysis/cna_predicted_vs_actual.png``

Usage
-----
    python plot_cna_reconstruction.py                                 # defaults
    LOSS_DIR=cna_loss_panel_filtered/ OUTPUT_DIR=plots/cna_analysis python plot_cna_reconstruction.py
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
LOSS_FILES_DIR = os.environ.get("LOSS_DIR", "cna_loss_panel_filtered/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "plots/cna_analysis")
ATTN_BLOCKS_TO_INCLUDE = (
    [int(x) for x in os.environ["ATTN_BLOCKS_TO_INCLUDE"].split(",")]
    if os.environ.get("ATTN_BLOCKS_TO_INCLUDE")
    else [0, 1, 2]
)

# When True, fix axes to +/-3 (z-score units) and use z-score axis labels.
# When False, axis range is derived from the data; if RAW_PLOT_LIM is set,
# axes are clamped to +/- that value.
Z_SCORE_NORM = os.environ.get("Z_SCORE_NORM", "0") != "0"
_raw_plot_lim_env = os.environ.get("RAW_PLOT_LIM")
RAW_PLOT_LIM = float(_raw_plot_lim_env) if _raw_plot_lim_env else None

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

    # Filename format: TCGA_PanCan_CNA_NoLOH_attn_<N>_msk_chord_loss_cna.pkl
    parts = pkl_file.split("_")
    attn_idx = int(parts[5])
    label = f"attn_{attn_idx}"
    data["cna_attention_blocks"] = attn_idx
    all_models_data[label] = data
    print(f"Loaded {label}")

model_order = sorted(all_models_data,
                     key=lambda x: all_models_data[x]["cna_attention_blocks"])
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

# Shared axis limits across panels (only when not z-scored and not user-clamped).
if not Z_SCORE_NORM and RAW_PLOT_LIM is None:
    mins, maxs = [], []
    for model in model_order:
        d = all_models_data[model]
        mins.extend([float(d["actual"].min()), float(d["predictions"].min())])
        maxs.extend([float(d["actual"].max()), float(d["predictions"].max())])
    global_min, global_max = min(mins), max(maxs)
    print(f"Shared axis range: [{global_min:.3f}, {global_max:.3f}]")
else:
    global_min = global_max = None

for idx, model in enumerate(model_order):
    data = all_models_data[model]
    preds = data["predictions"]
    actual = data["actual"]
    pearson_r, _ = stats.pearsonr(actual, preds)

    if len(actual) > SCATTER_MAX_POINTS:
        sub = np.random.choice(len(actual), SCATTER_MAX_POINTS, replace=False)
        plot_actual, plot_preds = actual[sub], preds[sub]
    else:
        plot_actual, plot_preds = actual, preds

    ax = axes[idx]
    ax.scatter(plot_actual, plot_preds, alpha=0.1, s=1, c=COLOR_PALETTE[model])

    if Z_SCORE_NORM:
        lim = 3.0
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=2)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        xlabel = "Actual Segment Mean (z-score)"
        ylabel = "Predicted Segment Mean (z-score)"
    elif RAW_PLOT_LIM is not None:
        lim = float(RAW_PLOT_LIM)
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=2)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        xlabel = "Actual Segment Mean (log2 ratio)"
        ylabel = "Predicted Segment Mean (log2 ratio)"
    else:
        ax.plot([global_min, global_max], [global_min, global_max],
                "r--", linewidth=2)
        ax.set_xlim([global_min, global_max])
        ax.set_ylim([global_min, global_max])
        xlabel = "Actual Segment Mean (log2 ratio)"
        ylabel = "Predicted Segment Mean (log2 ratio)"

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"attn_blocks = {data['cna_attention_blocks']}\nPearson r = {pearson_r:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")

plt.suptitle("CNA Segment Mean (MSK-CHORD)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
out_path = os.path.join(OUTPUT_DIR, "cna_predicted_vs_actual.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out_path}")
