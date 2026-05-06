"""Compare ROC + PR curves across the 7 SNV tumor-type classification models.

Loads per-model results saved by ``tumor_type_classifier_snv.py`` and
generates the cross-model comparison panels reported in Figure 3.

Inputs
------
``models_macro/<MODEL>_results.pkl`` for each variant model
(baseline / local_{1,10,25} / global_{1,10,25}).

Output
------
``plots/`` -- comparison ROC + PR figures.

Usage
-----
    python compare_models.py
"""

import glob
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
RESULTS_DIR = os.environ.get("RESULTS_DIR", "models_macro")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "plots")
MATPLOTLIB_BACKEND = os.environ.get("MATPLOTLIB_BACKEND")
if MATPLOTLIB_BACKEND:
    matplotlib.use(MATPLOTLIB_BACKEND, force=True)


# ============================================================================
# Load Results
# ============================================================================
print("=" * 60)
print("LOADING MODEL RESULTS")
print("=" * 60)

results_files = glob.glob(os.path.join(RESULTS_DIR, 'snv_ensemble_results_*.pkl'))

if not results_files:
    print(f"No results files found in {RESULTS_DIR}")
    print("Expected pattern: snv_ensemble_results_*.pkl")
    exit(1)

print(f"Found {len(results_files)} results files:")

all_results = {}
for fpath in sorted(results_files):
    with open(fpath, 'rb') as f:
        results = pickle.load(f)
    model_name = results['model_name']
    all_results[model_name] = results

    micro_auc = results['ensemble_performance']['micro_auc']
    macro_auc = results['ensemble_performance']['macro_auc']
    micro_ap = results['ensemble_performance']['micro_ap']
    macro_ap = results['ensemble_performance']['macro_ap']

    print(f"  {model_name}: AUC(micro={micro_auc:.3f}, macro={macro_auc:.3f}), "
          f"AP(micro={micro_ap:.3f}, macro={macro_ap:.3f})")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define colorblind-safe color palette (Paul Tol scheme - publication quality)
# Same scheme as tcga_pancan_snv/plot_accuracy.py
color_palette = {
    'baseline': '#777777',       # Gray baseline
    'baseline_loss': '#777777',  # Gray baseline (alternate name)
    'local_1':  '#88CCEE',       # Cyan (light)
    'global_1': '#0077BB',       # Blue (dark)
    'local_10':  '#DDAA33',      # Orange (light)
    'global_10': '#CC3311',      # Red (dark)
    'local_25':  '#AA3377',      # Magenta (light)
    'global_25': '#661100'       # Burgundy (dark)
}

# Preferred model order for consistent plot legends
model_order = ['baseline', 'baseline_loss', 'local_1', 'global_1', 'local_10', 'global_10', 'local_25', 'global_25']

# Sort models by preferred order, unknown models go to end
model_names = sorted(all_results.keys(), key=lambda x: model_order.index(x) if x in model_order else 999)


def get_color(model_name):
    """Get color for a model, with fallback for unknown models."""
    if model_name in color_palette:
        return color_palette[model_name]
    # Fallback color for unknown models
    return '#333333'


# ============================================================================
# Plot 1: Micro-Average ROC Comparison
# ============================================================================
print("\n" + "=" * 60)
print("CREATING ROC COMPARISON PLOTS")
print("=" * 60)

plt.figure(figsize=(8, 6))

for model_name in model_names:
    results = all_results[model_name]
    fpr = results['roc_data']['fpr']['micro']
    tpr = results['roc_data']['tpr']['micro']
    auc_val = results['roc_data']['roc_auc']['micro']

    plt.plot(fpr, tpr, color=get_color(model_name), lw=2,
             label=f'{model_name} (AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Micro-Average ROC Curves - Model Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_roc_micro.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/model_comparison_roc_micro.png")


# ============================================================================
# Plot 2: Macro-Average ROC Comparison
# ============================================================================
plt.figure(figsize=(8, 6))

for model_name in model_names:
    results = all_results[model_name]
    fpr = results['roc_data']['fpr']['macro']
    tpr = results['roc_data']['tpr']['macro']
    auc_val = results['roc_data']['roc_auc']['macro']

    plt.plot(fpr, tpr, color=get_color(model_name), lw=2,
             label=f'{model_name} (AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Macro-Average ROC Curves - Model Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_roc_macro.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/model_comparison_roc_macro.png")


# ============================================================================
# Plot 3: Micro-Average PR Comparison
# ============================================================================
print("\n" + "=" * 60)
print("CREATING PR COMPARISON PLOTS")
print("=" * 60)

plt.figure(figsize=(8, 6))

for model_name in model_names:
    results = all_results[model_name]
    precision = results['precision_recall_data']['precision']['micro']
    recall = results['precision_recall_data']['recall']['micro']
    ap_val = results['precision_recall_data']['average_precision']['micro']

    plt.plot(recall, precision, color=get_color(model_name), lw=2,
             label=f'{model_name} (AP = {ap_val:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Micro-Average Precision-Recall Curves - Model Comparison', fontsize=14)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_pr_micro.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/model_comparison_pr_micro.png")


# ============================================================================
# Plot 4: Macro-Average PR Comparison
# ============================================================================
plt.figure(figsize=(8, 6))

for model_name in model_names:
    results = all_results[model_name]
    precision = results['precision_recall_data']['precision']['macro']
    recall = results['precision_recall_data']['recall']['macro']
    ap_val = results['precision_recall_data']['average_precision']['macro']

    plt.plot(recall, precision, color=get_color(model_name), lw=2,
             label=f'{model_name} (AP = {ap_val:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Macro-Average Precision-Recall Curves - Model Comparison', fontsize=14)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_pr_macro.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/model_comparison_pr_macro.png")

plt.show()

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETE")
print("=" * 60)
