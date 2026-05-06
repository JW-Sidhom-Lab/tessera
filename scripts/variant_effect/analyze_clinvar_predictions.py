"""ClinVar pathogenicity-prediction analysis: ROC + PR figures and bootstrap CI metrics for Figure 1 h-o.

Reads per-model prediction CSVs written by
``generate_clinvar_predictions.py``, computes paired statistical tests
across models, and generates the publication-quality ROC + precision-
recall figures.

Models compared (same 7 as scripts/tcga_pancan_snv/): baseline,
local_{1,10,25}, global_{1,10,25}.

Inputs
------
``clinvar_preds_dedup_<strategy>/*_predictions.csv``  (from generate_clinvar_predictions.py)

Outputs
-------
``plots/clinvar_dedup_<strategy>/`` -- PDF / PNG figures.

Usage
-----
    python analyze_clinvar_predictions.py                            # variant strategy (default)
    STRATEGY=gene python analyze_clinvar_predictions.py
    REQUIRE_CORRECT_PREDICTIONS=0 python analyze_clinvar_predictions.py
"""

import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (bootstrap_metric, compute_metrics_with_ci,
                            flatten_metrics_dict, format_metric_for_display,
                            paired_statistical_test, save_plot, setup_plot_style)

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
MODELS = ['baseline', 'local_1', 'global_1', 'local_10', 'global_10',
          'local_25', 'global_25']

STRATEGY = os.environ.get("STRATEGY", "variant")  # 'variant' (Fig. 1 h-k) | 'gene' (Fig. 1 l-o)
PREDICTION_DIR = os.environ.get("PREDICTION_DIR", f"clinvar_preds_dedup_{STRATEGY}")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"plots/clinvar_dedup_{STRATEGY}")
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
CONFIDENCE_LEVEL = float(os.environ.get("CONFIDENCE_LEVEL", "0.95"))
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "baseline")
REQUIRE_CORRECT_PREDICTIONS = os.environ.get("REQUIRE_CORRECT_PREDICTIONS", "1") != "0"

# Color scheme (publication quality, colorblind-safe)
# Semantic grouping: baseline gray, context families with light/dark pairs
COLOR_PALETTE = {
    'baseline': '#777777',      # Gray baseline
    'local_1': '#88CCEE',       # Cyan (light)
    'global_1': '#0077BB',      # Blue (dark)
    'local_10': '#DDAA33',      # Orange (light)
    'global_10': '#CC3311',     # Red (dark)
    'local_25': '#AA3377',      # Magenta (light)
    'global_25': '#661100'      # Burgundy (dark)
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_clinvar_predictions(prediction_dir=PREDICTION_DIR, splits=None, require_correct_prediction=False):
    """
    Load all ClinVar prediction CSV files and prepare data.

    Parameters
    ----------
    prediction_dir : str
        Directory containing prediction CSV files
    splits : list, optional
        Splits to load (default: ['train', 'valid'])
    require_correct_prediction : bool, optional
        If True, filter to only variants correctly predicted by pre-training step
        (default: False, use all variants)

    Returns
    -------
    data : dict
        Structure: {split_name: {model_name: DataFrame}}
        Each DataFrame has columns:
        - chrom, start, ref, alt (variant ID)
        - clinvar_broad (pathogenic/benign/uncertain)
        - pred_proba (predicted probability of pathogenic)
        - label (binary: 1=pathogenic, 0=benign)
    """
    if splits is None:
        splits = ['train', 'valid']

    pred_dir = Path(prediction_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    data = {}

    for split in splits:
        data[split] = {}

        for model in MODELS:
            filepath = pred_dir / f'{model}_{split}_predictions.csv'

            if not filepath.exists():
                warnings.warn(f"File not found: {filepath}, skipping {model} {split}")
                continue

            try:
                df = pd.read_csv(filepath, low_memory=False)

                # Optionally filter to correctly predicted variants only
                if require_correct_prediction:
                    df = df[df['correctly_predicted'] == True].copy()

                # Filter to annotated variants
                df = df[df['CLNSIG'].notna()].copy()

                # Filter to pathogenic/benign (exclude uncertain)
                valid_sigs = ['Pathogenic', 'Benign', 'Likely_pathogenic', 'Likely_benign']
                df = df[df['CLNSIG'].isin(valid_sigs)].copy()

                if len(df) == 0:
                    warnings.warn(f"No labeled variants for {model} {split}")
                    continue

                # Create binary label (pathogenic = 1)
                pathogenic_values = ['Pathogenic', 'Likely_pathogenic']
                df['label'] = df['CLNSIG'].isin(pathogenic_values).astype(int)

                # Print summary
                n_pathogenic = (df['label'] == 1).sum()
                n_benign = (df['label'] == 0).sum()
                filter_note = " (correctly predicted only)" if require_correct_prediction else ""
                print(f"  {model:12s}: {len(df):6d} variants "
                      f"({n_pathogenic:5d} pathogenic, {n_benign:5d} benign){filter_note}")

                data[split][model] = df

            except Exception as e:
                warnings.warn(f"Error loading {filepath}: {e}")
                continue

    return data


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_all_metrics(data, n_bootstrap=N_BOOTSTRAP, splits=None):
    """
    Compute metrics for all models across all splits.

    Parameters
    ----------
    data : dict
        Data dictionary from load_clinvar_predictions()
    n_bootstrap : int
        Number of bootstrap iterations
    splits : list, optional
        Splits to analyze (default: all available)

    Returns
    -------
    metrics_df : pd.DataFrame
        Columns: [split, model, metric, value, ci_lower, ci_upper]
    """
    if splits is None:
        splits = list(data.keys())

    all_metrics = []

    for split in splits:
        print(f"\nComputing metrics for {split} split:")

        for model in MODELS:
            if model not in data[split]:
                continue

            df = data[split][model]
            y_true = df['label'].values
            # Column name is 'P(pathogenic)' from generate_clinvar_predictions.py
            y_pred_proba = df['P(pathogenic)'].values

            # Remove NaN predictions
            mask = ~np.isnan(y_pred_proba)
            if mask.sum() < len(y_pred_proba):
                warnings.warn(f"Removed {(~mask).sum()} NaN predictions for {model} {split}")
            y_true = y_true[mask]
            y_pred_proba = y_pred_proba[mask]

            if len(y_true) < 10:
                warnings.warn(f"Only {len(y_true)} samples for {model} {split}, skipping")
                continue

            try:
                metrics_dict = compute_metrics_with_ci(
                    y_true, y_pred_proba, threshold=0.5,
                    n_bootstrap=n_bootstrap, random_state=42
                )

                # Flatten to DataFrame rows
                rows = flatten_metrics_dict(metrics_dict, model, split)
                all_metrics.extend(rows)

                # Print summary for this model
                if 'ROC_AUC' in metrics_dict:
                    roc = metrics_dict['ROC_AUC']
                    print(f"  {model:12s}: ROC AUC = "
                          f"{roc['value']:.3f} [{roc['ci_lower']:.3f}, {roc['ci_upper']:.3f}]")

            except Exception as e:
                warnings.warn(f"Error computing metrics for {model} {split}: {e}")
                continue

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


# ============================================================================
# STATISTICAL COMPARISONS
# ============================================================================

def compare_models_statistically(data, test='wilcoxon', alpha=0.05,
                                 n_bootstrap=N_BOOTSTRAP, splits=None):
    """
    Perform all pairwise comparisons between models with statistical significance testing.

    Parameters
    ----------
    data : dict
        Data dictionary from load_clinvar_predictions()
    test : str
        Statistical test: 'wilcoxon' (default) or 'ttest'
    alpha : float
        Significance level
    n_bootstrap : int
        Bootstrap iterations
    splits : list, optional
        Splits to analyze

    Returns
    -------
    comparisons_df : pd.DataFrame
        Columns: [split, model_a, model_b, metric, value_a, value_b,
                  difference, p_value, significant, bonferroni_significant]
    """
    if splits is None:
        splits = list(data.keys())

    comparisons = []
    metrics_to_compare = ['ROC_AUC', 'PR_AUC', 'Balanced_Accuracy']

    for split in splits:
        print(f"\nPerforming all pairwise comparisons for {split} split:")

        # Get all model pairs
        model_list = [m for m in MODELS if m in data[split]]

        for i, model_a in enumerate(model_list):
            for model_b in model_list[i+1:]:  # Only compare each pair once
                df_a = data[split][model_a]
                y_true_a = df_a['label'].values
                y_pred_a = df_a['P(pathogenic)'].values

                # Remove NaN
                mask_a = ~np.isnan(y_pred_a)
                y_true_a = y_true_a[mask_a]
                y_pred_a = y_pred_a[mask_a]

                df_b = data[split][model_b]
                y_true_b = df_b['label'].values
                y_pred_b = df_b['P(pathogenic)'].values

                # Remove NaN
                mask_b = ~np.isnan(y_pred_b)
                y_true_b = y_true_b[mask_b]
                y_pred_b = y_pred_b[mask_b]

                # Ensure same number of samples for paired test
                if len(y_true_a) != len(y_true_b):
                    warnings.warn(f"Sample size mismatch for {model_a} vs {model_b} "
                                f"in {split} ({len(y_true_a)} vs {len(y_true_b)})")
                    continue

                if len(y_true_a) < 10:
                    warnings.warn(f"Insufficient samples for paired test ({len(y_true_a)})")
                    continue

                # Test each metric
                from sklearn.metrics import roc_auc_score, average_precision_score

                for metric_func, metric_name in [
                    (roc_auc_score, 'ROC_AUC'),
                    (average_precision_score, 'PR_AUC'),
                ]:
                    try:
                        result = paired_statistical_test(
                            y_true_a, y_pred_a, y_pred_b,
                            metric_func, test=test, n_bootstrap=n_bootstrap
                        )

                        comparisons.append({
                            'split': split,
                            'model_a': model_a,
                            'model_b': model_b,
                            'metric': metric_name,
                            'value_a': result['metric_a'],
                            'value_b': result['metric_b'],
                            'difference': result['difference'],
                            'p_value': result['p_value'],
                            'significant': result['significant']
                        })

                    except Exception as e:
                        warnings.warn(f"Error comparing {model_a} to {model_b} "
                                    f"for {metric_name} in {split}: {e}")

                # Balanced accuracy (custom function)
                try:
                    def balanced_acc(y_true, y_pred):
                        from sklearn.metrics import balanced_accuracy_score
                        y_pred_binary = (y_pred >= 0.5).astype(int)
                        return balanced_accuracy_score(y_true, y_pred_binary)

                    result = paired_statistical_test(
                        y_true_a, y_pred_a, y_pred_b,
                        balanced_acc, test=test, n_bootstrap=n_bootstrap
                    )

                    comparisons.append({
                        'split': split,
                        'model_a': model_a,
                        'model_b': model_b,
                        'metric': 'Balanced_Accuracy',
                        'value_a': result['metric_a'],
                        'value_b': result['metric_b'],
                        'difference': result['difference'],
                        'p_value': result['p_value'],
                        'significant': result['significant']
                    })

                except Exception as e:
                    warnings.warn(f"Error computing balanced accuracy for {model_a} vs {model_b}: {e}")

    # Create DataFrame
    comparisons_df = pd.DataFrame(comparisons)

    # Apply Bonferroni correction
    if len(comparisons_df) > 0:
        n_tests = len(comparisons_df)
        corrected_alpha = alpha / n_tests
        comparisons_df['bonferroni_significant'] = comparisons_df['p_value'] < corrected_alpha
        comparisons_df['bonferroni_alpha'] = corrected_alpha

        # Print results with significance markers
        for split in comparisons_df['split'].unique():
            split_data = comparisons_df[comparisons_df['split'] == split]
            print(f"\n{split.upper()} SPLIT - Pairwise Comparisons:")
            for _, row in split_data.iterrows():
                sig_marker = "***" if row['bonferroni_significant'] else \
                             ("**" if row['significant'] else " ")
                print(f"  {row['model_a']:12s} vs {row['model_b']:12s} ({row['metric']:18s}): "
                      f"Δ = {row['difference']:+.4f}, p = {row['p_value']:.4f} {sig_marker}")

    return comparisons_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_roc_curves(data, output_dir=OUTPUT_DIR, split='valid', require_correct_prediction=False, metrics_df=None):
    """
    Plot ROC curves for all models.

    Parameters
    ----------
    data : dict
        Data dictionary from load_clinvar_predictions()
    output_dir : str
        Output directory for plots
    split : str
        Data split to plot
    require_correct_prediction : bool
        Whether the data was filtered to correctly predicted variants only
    metrics_df : pd.DataFrame, optional
        Precomputed metrics dataframe to avoid redundant bootstrap computation
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    setup_plot_style(fontsize=11)
    fig, ax = plt.subplots(figsize=(8, 7))

    if split not in data:
        warnings.warn(f"Split '{split}' not found in data")
        return

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')

    # Build lookup dictionary for metrics (much faster than repeated filtering)
    metrics_lookup = {}
    if metrics_df is not None:
        for _, row in metrics_df[metrics_df['metric'] == 'ROC_AUC'].iterrows():
            key = (row['split'], row['model'])
            metrics_lookup[key] = (row['value'], row['ci_lower'], row['ci_upper'])

    # Plot each model with COLOR_PALETTE
    for model in MODELS:
        color = COLOR_PALETTE.get(model, '#808080')
        if model not in data[split]:
            continue

        df = data[split][model]
        y_true = df['label'].values
        y_pred_proba = df['P(pathogenic)'].values

        # Remove NaN
        mask = ~np.isnan(y_pred_proba)
        y_true = y_true[mask]
        y_pred_proba = y_pred_proba[mask]

        # Get ROC AUC and CI
        try:
            # Try to get precomputed metrics (faster)
            lookup_key = (split, model)
            if lookup_key in metrics_lookup:
                auc, auc_ci_low, auc_ci_high = metrics_lookup[lookup_key]
            else:
                # Fallback: compute with reduced bootstrap if metrics not found
                auc, auc_ci_low, auc_ci_high = bootstrap_metric(
                    y_true, y_pred_proba, roc_auc_score, n_bootstrap=100
                )

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            label = f"{model} (AUC={format_metric_for_display(auc, auc_ci_low, auc_ci_high)})"
            ax.plot(fpr, tpr, lw=2.5, color=color, label=label)

        except Exception as e:
            warnings.warn(f"Error plotting ROC for {model}: {e}")

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    title = f'ROC Curves - ClinVar Pathogenicity ({split.upper()} split)'
    if require_correct_prediction:
        title += '\n(Correctly Predicted Only)'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3, linewidth=0.5)

    output_path = Path(output_dir) / f'roc_curves_{split}'
    save_plot(fig, str(output_path))
    plt.close(fig)


def plot_pr_curves(data, output_dir=OUTPUT_DIR, split='valid', require_correct_prediction=False, metrics_df=None):
    """
    Plot Precision-Recall curves for all models.

    Parameters
    ----------
    data : dict
        Data dictionary from load_clinvar_predictions()
    output_dir : str
        Output directory for plots
    split : str
        Data split to plot
    require_correct_prediction : bool
        Whether the data was filtered to correctly predicted variants only
    metrics_df : pd.DataFrame, optional
        Precomputed metrics dataframe to avoid redundant bootstrap computation
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    setup_plot_style(fontsize=11)
    fig, ax = plt.subplots(figsize=(8, 7))

    if split not in data:
        warnings.warn(f"Split '{split}' not found in data")
        return

    # Build lookup dictionary for metrics (much faster than repeated filtering)
    metrics_lookup = {}
    if metrics_df is not None:
        for _, row in metrics_df[metrics_df['metric'] == 'PR_AUC'].iterrows():
            key = (row['split'], row['model'])
            metrics_lookup[key] = (row['value'], row['ci_lower'], row['ci_upper'])

    # Plot each model with COLOR_PALETTE
    for model in MODELS:
        color = COLOR_PALETTE.get(model, '#808080')
        if model not in data[split]:
            continue

        df = data[split][model]
        y_true = df['label'].values
        y_pred_proba = df['P(pathogenic)'].values

        # Remove NaN
        mask = ~np.isnan(y_pred_proba)
        y_true = y_true[mask]
        y_pred_proba = y_pred_proba[mask]

        try:
            # Try to get precomputed metrics (faster)
            lookup_key = (split, model)
            if lookup_key in metrics_lookup:
                ap, ap_ci_low, ap_ci_high = metrics_lookup[lookup_key]
            else:
                # Fallback: compute with reduced bootstrap if metrics not found
                ap, ap_ci_low, ap_ci_high = bootstrap_metric(
                    y_true, y_pred_proba, average_precision_score, n_bootstrap=100
                )

            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            label = f"{model} (AP={format_metric_for_display(ap, ap_ci_low, ap_ci_high)})"
            ax.plot(recall, precision, lw=2.5, color=color, label=label)

        except Exception as e:
            warnings.warn(f"Error plotting PR for {model}: {e}")

    # Prevalence line
    prevalence = (y_true == 1).mean() if len(y_true) > 0 else 0.5
    ax.axhline(y=prevalence, color='k', linestyle='--', lw=1.5, alpha=0.5,
               label=f'Prevalence ({prevalence:.3f})')

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    title = f'Precision-Recall Curves - ClinVar Pathogenicity ({split.upper()} split)'
    if require_correct_prediction:
        title += '\n(Correctly Predicted Only)'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3, linewidth=0.5)

    output_path = Path(output_dir) / f'pr_curves_{split}'
    save_plot(fig, str(output_path))
    plt.close(fig)


def plot_metrics_comparison(metrics_df, output_dir=OUTPUT_DIR, split='valid', require_correct_prediction=False):
    """
    Plot metrics comparison with error bars using consistent color palette.

    Displays 2x3 grid showing ROC AUC, PR AUC, Balanced Accuracy,
    Precision (pathogenic), Recall (pathogenic), F1 (pathogenic) with 95% CI.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe from compute_all_metrics()
    output_dir : str
        Output directory for plots
    split : str
        Data split to plot
    require_correct_prediction : bool
        Whether the data was filtered to correctly predicted variants only
    """
    setup_plot_style(fontsize=11)

    # Filter to requested split
    split_data = metrics_df[metrics_df['split'] == split].copy()
    if len(split_data) == 0:
        warnings.warn(f"No metrics data for split '{split}'")
        return

    # Metrics to plot in order with display labels
    metrics_to_plot = ['ROC_AUC', 'PR_AUC', 'Balanced_Accuracy',
                       'Precision_1', 'Recall_1', 'F1_1']
    metric_labels = {
        'ROC_AUC': 'ROC AUC',
        'PR_AUC': 'PR AUC',
        'Balanced_Accuracy': 'Balanced Accuracy',
        'Precision_1': 'Precision (Pathogenic)',
        'Recall_1': 'Recall (Pathogenic)',
        'F1_1': 'F1 (Pathogenic)'
    }

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        metric_data = split_data[split_data['metric'] == metric].copy()

        if len(metric_data) == 0:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            continue

        # Sort by model order
        metric_data['model_order'] = metric_data['model'].map(
            {model: i for i, model in enumerate(MODELS)}
        )
        metric_data = metric_data.sort_values('model_order')

        x_pos = np.arange(len(metric_data))
        values = metric_data['value'].values
        errors = np.array([
            values - metric_data['ci_lower'].values,
            metric_data['ci_upper'].values - values
        ])

        # Use COLOR_PALETTE for each model
        colors = [COLOR_PALETTE.get(model, '#808080') for model in metric_data['model'].values]

        # Create bar chart with error bars
        ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.85,
               color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on top of bars
        for i, (v, e_upper) in enumerate(zip(values, errors[1])):
            ax.text(i, v + e_upper + 0.02, f'{v:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_data['model'].values, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric_labels.get(metric, metric.replace('_', ' ')), fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)

    # Add figure title with filter status
    suptitle = f'ClinVar Pathogenicity - Metrics Comparison ({split.upper()} split)'
    if require_correct_prediction:
        suptitle += '\n(Correctly Predicted Only)'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    output_path = Path(output_dir) / f'metrics_comparison_{split}'
    save_plot(fig, str(output_path))
    plt.close(fig)




# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(splits=None, n_bootstrap=N_BOOTSTRAP, output_dir=OUTPUT_DIR,
         prediction_dir=PREDICTION_DIR, require_correct_prediction=False):
    """
    Main analysis pipeline.

    Parameters
    ----------
    splits : list, optional
        Splits to analyze (default: ['train', 'valid'])
    n_bootstrap : int
        Bootstrap iterations
    output_dir : str
        Output directory for results
    prediction_dir : str
        Directory with prediction CSVs
    require_correct_prediction : bool, optional
        If True, filter to only variants correctly predicted by pre-training step
        (default: False, use all variants)
    """
    if splits is None:
        splits = ['train', 'valid']

    # Append suffix to output directory if filtering
    if require_correct_prediction:
        output_dir = output_dir.rstrip('/') + '_correct_only'

    print("="*80)
    print("ClinVar Pathogenicity Prediction Analysis")
    if require_correct_prediction:
        print("(Filtering to correctly predicted variants only)")
    print("="*80)

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    overall_start = time.perf_counter()

    # Load predictions
    print("1. Loading ClinVar predictions...")
    step_start = time.perf_counter()
    data = load_clinvar_predictions(prediction_dir, splits,
                                   require_correct_prediction=require_correct_prediction)
    step_time = time.perf_counter() - step_start
    print(f"   ✓ Completed in {step_time:.2f}s\n")

    # Compute metrics
    print("2. Computing metrics with bootstrap confidence intervals (N_BOOTSTRAP={})...".format(n_bootstrap))
    step_start = time.perf_counter()
    metrics_df = compute_all_metrics(data, n_bootstrap=n_bootstrap, splits=splits)
    step_time = time.perf_counter() - step_start
    print(f"   ✓ Completed in {step_time:.2f}s")

    if len(metrics_df) > 0:
        metrics_path = Path(output_dir) / 'metrics_summary.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"   Saved: {metrics_path}\n")

    # Statistical comparisons
    print("3. Performing pairwise comparisons with statistical tests...")
    step_start = time.perf_counter()
    comparisons_df = compare_models_statistically(
        data, n_bootstrap=n_bootstrap, splits=splits
    )
    step_time = time.perf_counter() - step_start
    print(f"   ✓ Completed in {step_time:.2f}s")

    if len(comparisons_df) > 0:
        comparisons_path = Path(output_dir) / 'statistical_comparisons.csv'
        comparisons_df.to_csv(comparisons_path, index=False)
        print(f"   Saved: {comparisons_path}\n")

    # Generate plots
    print("4. Generating plots...")
    plots_start = time.perf_counter()

    for split in splits:
        if split not in data or len(data[split]) == 0:
            continue

        print(f"\n   {split.upper()} split:")

        step_start = time.perf_counter()
        plot_roc_curves(data, output_dir, split, require_correct_prediction=require_correct_prediction, metrics_df=metrics_df)
        print(f"     ✓ ROC curves ({time.perf_counter() - step_start:.2f}s)")

        step_start = time.perf_counter()
        plot_pr_curves(data, output_dir, split, require_correct_prediction=require_correct_prediction, metrics_df=metrics_df)
        print(f"     ✓ PR curves ({time.perf_counter() - step_start:.2f}s)")

        if len(metrics_df) > 0:
            step_start = time.perf_counter()
            plot_metrics_comparison(metrics_df, output_dir, split, require_correct_prediction=require_correct_prediction)
            print(f"     ✓ Metrics comparison ({time.perf_counter() - step_start:.2f}s)")

    plots_time = time.perf_counter() - plots_start
    print(f"\n   Total plotting time: {plots_time:.2f}s")

    overall_time = time.perf_counter() - overall_start
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Total runtime: {overall_time:.2f}s ({overall_time/60:.1f}m)")
    print("="*80)


if __name__ == '__main__':
    # Run from PyCharm - modify parameters as needed
    main(
        splits=['train', 'valid'],
        n_bootstrap=N_BOOTSTRAP,
        output_dir=OUTPUT_DIR,
        prediction_dir=PREDICTION_DIR,
        require_correct_prediction=REQUIRE_CORRECT_PREDICTIONS,
    )
