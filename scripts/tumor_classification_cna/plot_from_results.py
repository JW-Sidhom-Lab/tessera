"""Per-model plots from saved CNA tumor-type classification results.

Reads a single model's results pkl and emits the per-model figure
panels in Figure 3. Available models: attn_0, attn_1, attn_2.

Inputs
------
``<RESULTS_DIR>/cna_ensemble_results_<MODEL>.pkl`` (from
``tumor_type_classifier_cna.py``).

Output
------
``<PLOTS_DIR>/`` -- per-model figure files.

Usage
-----
    python plot_from_results.py                  # attn_2 (default)
    MODEL_NAME=attn_0 python plot_from_results.py
"""

import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
MODEL_NAME = os.environ.get("MODEL_NAME", "attn_2")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "models_macro")
PLOTS_DIR = os.environ.get("PLOTS_DIR", "plots")
MATPLOTLIB_BACKEND = os.environ.get("MATPLOTLIB_BACKEND")
if MATPLOTLIB_BACKEND:
    matplotlib.use(MATPLOTLIB_BACKEND, force=True)


# ============================================================================
# Fixed Color Palette for Tumor Types (Consistent Across All Plots)
# 23 tumor types with COAD+READ → COADREAD and ESCA+STAD → ESCASTAD
# ============================================================================
TUMOR_TYPE_COLORS = {
    'BLCA': '#88CCEE',     # Cyan
    'BRCA': '#44AA99',     # Teal
    'CESC': '#117733',     # Green
    'COADREAD': '#DDCC77', # Sand
    'ESCASTAD': '#882255', # Wine
    'GBM': '#AA4499',      # Purple
    'HNSC': '#661100',     # Brown
    'KIRC': '#0077BB',     # Blue
    'KIRP': '#33BBEE',     # Light blue
    'LGG': '#EE7733',      # Orange
    'LIHC': '#CC3311',     # Red
    'LUAD': '#EE3377',     # Magenta
    'LUSC': '#004488',     # Dark blue
    'OV': '#44BB99',       # Mint
    'PAAD': '#99DDFF',     # Light cyan
    'PCPG': '#77AADD',     # Periwinkle
    'PRAD': '#EE8866',     # Coral
    'SARC': '#FFAABB',     # Pink
    'SKCM': '#332288',     # Dark purple
    'TGCT': '#009988',     # Teal green
    'THCA': '#6699CC',     # Steel blue
    'THYM': '#BBCC33',     # Lime
    'UCEC': '#AAAA00',     # Dark yellow
}

FALLBACK_COLORS = [
    '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
    '#DDCC77', '#CC6677', '#882255', '#AA4499', '#661100',
    '#6699CC', '#0077BB', '#33BBEE', '#009988', '#EE7733',
    '#CC3311', '#EE3377', '#004488', '#BBBBBB', '#44BB99',
]


def get_tumor_color(tumor_type, index=0):
    """Get consistent color for a tumor type."""
    if tumor_type in TUMOR_TYPE_COLORS:
        return TUMOR_TYPE_COLORS[tumor_type]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def plot_from_results(model_name, results_dir=RESULTS_DIR, output_dir=PLOTS_DIR):
    """
    Load results from pickle file and generate all plots.

    Args:
        model_name: Name of the CNA model (e.g., 'attn_0', 'attn_1', 'attn_2')
        results_dir: Directory containing results .pkl files
        output_dir: Output directory for plots
    """
    # Construct results file path
    results_path = os.path.join(results_dir, f'cna_ensemble_results_{model_name}.pkl')

    if not os.path.exists(results_path):
        print(f"Error: Results file not found: {results_path}")
        print(f"Available models in {results_dir}:")
        import glob
        for f in glob.glob(os.path.join(results_dir, 'cna_ensemble_results_*.pkl')):
            name = os.path.basename(f).replace('cna_ensemble_results_', '').replace('.pkl', '')
            print(f"  - {name}")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    print(f"Model name: {model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    roc_data = results['roc_data']
    pr_data = results['precision_recall_data']
    cm_data = results['confusion_matrix']

    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    roc_auc = roc_data['roc_auc']

    precision = pr_data['precision']
    recall = pr_data['recall']
    average_precision = pr_data['average_precision']

    cm_all = cm_data['raw']
    cm_normalized = cm_data['normalized']
    tumor_type_names = cm_data['class_names']

    all_class_indices = [i for i in roc_auc.keys() if isinstance(i, int)]
    n_classes = len(all_class_indices)

    print(f"Number of classes: {n_classes}")
    print(f"Tumor types: {list(tumor_type_names)}")
    print(f"Micro AUC: {roc_auc['micro']:.4f}")
    print(f"Macro AUC: {roc_auc['macro']:.4f}")
    print(f"Micro AP: {average_precision['micro']:.4f}")
    print(f"Macro AP: {average_precision['macro']:.4f}")

    colors = [get_tumor_color(tumor_type_names[i], i) for i in all_class_indices]

    # Plot 1: ROC Curves
    print("\nGenerating ROC curves plot...")
    plt.figure(figsize=(12, 9))

    for i, color in zip(all_class_indices, colors):
        class_name = tumor_type_names[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.8,
                 label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - CNA Ensemble - {model_name}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(output_dir, f'cna_ensemble_roc_curves_{model_name}.png')
    plt.savefig(roc_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {roc_path}")

    # Plot 2: Precision-Recall Curves
    print("Generating PR curves plot...")
    plt.figure(figsize=(12, 9))

    for i, color in zip(all_class_indices, colors):
        class_name = tumor_type_names[i]
        plt.plot(recall[i], precision[i], color=color, lw=1.5, alpha=0.8,
                 label=f'{class_name} (AP = {average_precision[i]:.3f})')

    plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average (AP = {average_precision["micro"]:.3f})')
    plt.plot(recall["macro"], precision["macro"], color='navy', linestyle=':', linewidth=4,
             label=f'Macro-average (AP = {average_precision["macro"]:.3f})')

    n_samples = np.sum(cm_all)
    class_counts = np.sum(cm_all, axis=1)
    baseline_precision = class_counts / n_samples
    plt.axhline(y=baseline_precision.mean(), color='k', linestyle='--', lw=2, alpha=0.8,
                label=f'Random baseline (AP = {baseline_precision.mean():.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - CNA Ensemble - {model_name}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pr_path = os.path.join(output_dir, f'cna_ensemble_precision_recall_{model_name}.png')
    plt.savefig(pr_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {pr_path}")

    # Plot 3: Confusion Matrix (Normalized)
    print("Generating confusion matrix (normalized)...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=tumor_type_names, yticklabels=tumor_type_names,
                cbar_kws={'label': 'Normalized Frequency (%)'}, annot_kws={"size": 8})

    plt.title(f'Confusion Matrix - CNA Ensemble (Normalized) - {model_name}', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_norm_path = os.path.join(output_dir, f'cna_ensemble_confusion_matrix_normalized_{model_name}.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {cm_norm_path}")

    # Plot 4: Confusion Matrix (Raw Counts)
    print("Generating confusion matrix (raw counts)...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues',
                xticklabels=tumor_type_names, yticklabels=tumor_type_names,
                cbar_kws={'label': 'Count'})

    plt.title(f'Confusion Matrix - CNA Ensemble (Raw Counts) - {model_name}', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_raw_path = os.path.join(output_dir, f'cna_ensemble_confusion_matrix_counts_{model_name}.png')
    plt.savefig(cm_raw_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {cm_raw_path}")

    plt.show()
    print("\nAll plots generated successfully!")
    return results


plot_from_results(MODEL_NAME, RESULTS_DIR, PLOTS_DIR)
