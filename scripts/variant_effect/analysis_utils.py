"""
Shared utilities for variant effect prediction analysis.

Provides statistical testing, metrics computation with confidence intervals,
and plotting utilities for manuscript-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import warnings
import os


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_metric(y_true, y_pred_proba, metric_func, n_bootstrap=1000, random_state=42):
    """
    Compute metric value with 95% bootstrap confidence interval.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for class 1
    metric_func : callable
        Function to compute metric (e.g., roc_auc_score)
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    metric_value : float
        Computed metric on full data
    ci_lower : float
        Lower bound of 95% CI (2.5th percentile)
    ci_upper : float
        Upper bound of 95% CI (97.5th percentile)
    """
    np.random.seed(random_state)

    n_samples = len(y_true)
    bootstrap_metrics = []
    valid_iterations = 0

    for _ in range(n_bootstrap):
        # Bootstrap resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            metric = metric_func(y_true_boot, y_pred_boot)
            if not np.isnan(metric) and not np.isinf(metric):
                bootstrap_metrics.append(metric)
                valid_iterations += 1
        except Exception:
            continue

    if valid_iterations < 100:
        warnings.warn(
            f"Only {valid_iterations} valid bootstrap iterations (expected ~{n_bootstrap}). "
            "CI may be unreliable."
        )

    if len(bootstrap_metrics) == 0:
        raise ValueError("No valid bootstrap iterations")

    # Compute metric on full data
    metric_value = metric_func(y_true, y_pred_proba)

    # Percentile method for CI
    bootstrap_metrics = np.array(bootstrap_metrics)
    ci_lower = np.percentile(bootstrap_metrics, 2.5)
    ci_upper = np.percentile(bootstrap_metrics, 97.5)

    # Validate CI
    if not (ci_lower <= metric_value <= ci_upper):
        warnings.warn(
            f"Metric value {metric_value:.3f} outside CI [{ci_lower:.3f}, {ci_upper:.3f}]. "
            "This can happen with small samples or skewed distributions."
        )

    return metric_value, ci_lower, ci_upper


# ============================================================================
# METRICS COMPUTATION WITH CONFIDENCE INTERVALS
# ============================================================================

def compute_metrics_with_ci(y_true, y_pred_proba, threshold=0.5, n_bootstrap=1000, random_state=42):
    """
    Compute comprehensive metrics with 95% bootstrap confidence intervals.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for class 1
    threshold : float
        Classification threshold for binary predictions
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed

    Returns
    -------
    metrics_dict : dict
        Dictionary with metrics and CIs:
        {
            'ROC_AUC': {'value': ..., 'ci_lower': ..., 'ci_upper': ...},
            'PR_AUC': {...},
            'Accuracy': {...},
            'Balanced_Accuracy': {...},
            'Precision_0': {...},
            'Recall_0': {...},
            'F1_0': {...},
            'Precision_1': {...},
            'Recall_1': {...},
            'F1_1': {...},
            'Confusion_Matrix': array([[TN, FP], [FN, TP]])
        }
    """
    metrics_dict = {}

    # Make binary predictions
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    # ROC AUC
    try:
        roc_auc, roc_ci_low, roc_ci_high = bootstrap_metric(
            y_true, y_pred_proba, roc_auc_score, n_bootstrap=n_bootstrap, random_state=random_state
        )
        metrics_dict['ROC_AUC'] = {
            'value': roc_auc,
            'ci_lower': roc_ci_low,
            'ci_upper': roc_ci_high
        }
    except Exception as e:
        warnings.warn(f"Could not compute ROC AUC: {e}")

    # PR AUC (Average Precision)
    try:
        pr_auc, pr_ci_low, pr_ci_high = bootstrap_metric(
            y_true, y_pred_proba, average_precision_score, n_bootstrap=n_bootstrap, random_state=random_state
        )
        metrics_dict['PR_AUC'] = {
            'value': pr_auc,
            'ci_lower': pr_ci_low,
            'ci_upper': pr_ci_high
        }
    except Exception as e:
        warnings.warn(f"Could not compute PR AUC: {e}")

    # Accuracy
    def accuracy_from_proba(y_true, y_pred_proba):
        return accuracy_score(y_true, (y_pred_proba >= threshold).astype(int))

    try:
        acc, acc_ci_low, acc_ci_high = bootstrap_metric(
            y_true, y_pred_proba, accuracy_from_proba, n_bootstrap=n_bootstrap, random_state=random_state
        )
        metrics_dict['Accuracy'] = {
            'value': acc,
            'ci_lower': acc_ci_low,
            'ci_upper': acc_ci_high
        }
    except Exception as e:
        warnings.warn(f"Could not compute Accuracy: {e}")

    # Balanced Accuracy
    def balanced_acc_from_proba(y_true, y_pred_proba):
        return balanced_accuracy_score(y_true, (y_pred_proba >= threshold).astype(int))

    try:
        bal_acc, bal_acc_ci_low, bal_acc_ci_high = bootstrap_metric(
            y_true, y_pred_proba, balanced_acc_from_proba, n_bootstrap=n_bootstrap, random_state=random_state
        )
        metrics_dict['Balanced_Accuracy'] = {
            'value': bal_acc,
            'ci_lower': bal_acc_ci_low,
            'ci_upper': bal_acc_ci_high
        }
    except Exception as e:
        warnings.warn(f"Could not compute Balanced Accuracy: {e}")

    # Per-class metrics (Precision, Recall, F1)
    for class_label in [0, 1]:
        class_name = {0: '_0', 1: '_1'}[class_label]

        # Precision
        try:
            def precision_func(y_true, y_pred_proba):
                y_pred = (y_pred_proba >= threshold).astype(int)
                return precision_score(y_true, y_pred, labels=[class_label], zero_division=0)

            prec, prec_ci_low, prec_ci_high = bootstrap_metric(
                y_true, y_pred_proba, precision_func, n_bootstrap=n_bootstrap, random_state=random_state
            )
            metrics_dict[f'Precision{class_name}'] = {
                'value': prec,
                'ci_lower': prec_ci_low,
                'ci_upper': prec_ci_high
            }
        except Exception as e:
            warnings.warn(f"Could not compute Precision for class {class_label}: {e}")

        # Recall
        try:
            def recall_func(y_true, y_pred_proba):
                y_pred = (y_pred_proba >= threshold).astype(int)
                return recall_score(y_true, y_pred, labels=[class_label], zero_division=0)

            rec, rec_ci_low, rec_ci_high = bootstrap_metric(
                y_true, y_pred_proba, recall_func, n_bootstrap=n_bootstrap, random_state=random_state
            )
            metrics_dict[f'Recall{class_name}'] = {
                'value': rec,
                'ci_lower': rec_ci_low,
                'ci_upper': rec_ci_high
            }
        except Exception as e:
            warnings.warn(f"Could not compute Recall for class {class_label}: {e}")

        # F1
        try:
            def f1_func(y_true, y_pred_proba):
                y_pred = (y_pred_proba >= threshold).astype(int)
                return f1_score(y_true, y_pred, labels=[class_label], zero_division=0)

            f1, f1_ci_low, f1_ci_high = bootstrap_metric(
                y_true, y_pred_proba, f1_func, n_bootstrap=n_bootstrap, random_state=random_state
            )
            metrics_dict[f'F1{class_name}'] = {
                'value': f1,
                'ci_lower': f1_ci_low,
                'ci_upper': f1_ci_high
            }
        except Exception as e:
            warnings.warn(f"Could not compute F1 for class {class_label}: {e}")

    # Confusion Matrix (on full data)
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    metrics_dict['Confusion_Matrix'] = cm

    return metrics_dict


# ============================================================================
# PAIRED STATISTICAL TESTS
# ============================================================================

def paired_statistical_test(y_true, y_pred_proba_a, y_pred_proba_b, metric_func,
                           test='wilcoxon', n_bootstrap=1000, random_state=42):
    """
    Perform paired statistical test between two models.

    Parameters
    ----------
    y_true : array-like
        True binary labels (must be same length for both models)
    y_pred_proba_a : array-like
        Predictions from model A
    y_pred_proba_b : array-like
        Predictions from model B
    metric_func : callable
        Metric to compare (e.g., roc_auc_score)
    test : str
        'wilcoxon' (default, non-parametric) or 'ttest' (parametric)
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed

    Returns
    -------
    result : dict
        {
            'metric_a': float,
            'metric_b': float,
            'difference': float (metric_b - metric_a),
            'p_value': float,
            'significant': bool (p < 0.05)
        }
    """
    np.random.seed(random_state)

    # Compute metrics on full data
    metric_a = metric_func(y_true, y_pred_proba_a)
    metric_b = metric_func(y_true, y_pred_proba_b)

    # Bootstrap to get metric distributions
    n_samples = len(y_true)
    diff_list = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_proba_a[indices]
        y_pred_b_boot = y_pred_proba_b[indices]

        # Skip if only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            m_a = metric_func(y_true_boot, y_pred_a_boot)
            m_b = metric_func(y_true_boot, y_pred_b_boot)

            if not np.isnan(m_a) and not np.isnan(m_b) and not np.isinf(m_a) and not np.isinf(m_b):
                diff_list.append(m_b - m_a)
        except Exception:
            continue

    if len(diff_list) < 50:
        warnings.warn(
            f"Only {len(diff_list)} valid bootstrap pairs. "
            "Statistical test may be unreliable."
        )
        return {
            'metric_a': metric_a,
            'metric_b': metric_b,
            'difference': metric_b - metric_a,
            'p_value': np.nan,
            'significant': False
        }

    diff_array = np.array(diff_list)

    # Perform statistical test
    if test == 'wilcoxon':
        # Wilcoxon signed-rank test (non-parametric)
        # H0: differences are symmetric around zero
        try:
            statistic, p_value = stats.wilcoxon(diff_array, alternative='two-sided')
        except Exception as e:
            warnings.warn(f"Wilcoxon test failed: {e}")
            p_value = np.nan

    elif test == 'ttest':
        # Paired t-test (parametric)
        # H0: mean difference is zero
        try:
            statistic, p_value = stats.ttest_1samp(diff_array, 0)
        except Exception as e:
            warnings.warn(f"t-test failed: {e}")
            p_value = np.nan
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        'metric_a': metric_a,
        'metric_b': metric_b,
        'difference': metric_b - metric_a,
        'p_value': p_value,
        'significant': p_value < 0.05 if not np.isnan(p_value) else False
    }


# ============================================================================
# MULTIPLE COMPARISONS CORRECTION
# ============================================================================

def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : array-like
        Array of p-values
    alpha : float
        Significance level

    Returns
    -------
    significant : array-like
        Boolean array indicating significance after correction
    corrected_alpha : float
        Adjusted alpha level: alpha / n_tests
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = p_values < corrected_alpha

    return significant, corrected_alpha


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def setup_plot_style(fontsize=11):
    """
    Configure matplotlib for publication-quality plots.

    Parameters
    ----------
    fontsize : int
        Base font size in points
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize + 2,
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'legend.fontsize': fontsize - 1,
        'figure.titlesize': fontsize + 3,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_palette("husl")


def save_plot(fig, filepath, formats=['png'], dpi=600):
    """
    Save plot in specified formats consistently.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str
        Path without extension (e.g., 'plots/roc_curves')
    formats : list
        Formats to save (default: ['png'])
    dpi : int
        DPI for PNG (default: 600 for publication quality)
    """
    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for fmt in formats:
        if fmt == 'png':
            fig.savefig(f"{filepath}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
        elif fmt == 'pdf':
            # Note: fonttype parameter may not be supported in all matplotlib versions
            try:
                fig.savefig(f"{filepath}.pdf", bbox_inches='tight', facecolor='white', fonttype=42)
            except TypeError:
                # Fallback for newer matplotlib that doesn't support fonttype
                fig.savefig(f"{filepath}.pdf", bbox_inches='tight', facecolor='white')
        else:
            warnings.warn(f"Unknown format: {fmt}")

    ext_str = ",".join([f".{fmt}" for fmt in formats])
    print(f"Saved plot to {filepath}{{{ext_str}}}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def flatten_metrics_dict(metrics_dict, model_name, split, variant_ids=None):
    """
    Convert metrics dictionary to DataFrame row for easier manipulation.

    Parameters
    ----------
    metrics_dict : dict
        Output from compute_metrics_with_ci()
    model_name : str
        Name of model
    split : str
        Train/valid split name
    variant_ids : array-like, optional
        Variant identifiers (for linking to raw data)

    Returns
    -------
    rows : list of dict
        List of dictionaries suitable for DataFrame construction
    """
    rows = []

    for metric_name, metric_value in metrics_dict.items():
        if metric_name == 'Confusion_Matrix':
            continue  # Skip confusion matrix for now

        if isinstance(metric_value, dict) and 'value' in metric_value:
            rows.append({
                'model': model_name,
                'split': split,
                'metric': metric_name,
                'value': metric_value['value'],
                'ci_lower': metric_value['ci_lower'],
                'ci_upper': metric_value['ci_upper']
            })

    return rows


def format_metric_for_display(value, ci_lower, ci_upper, decimals=3):
    """
    Format metric value with CI for display.

    Parameters
    ----------
    value : float
        Metric value
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound
    decimals : int
        Number of decimal places

    Returns
    -------
    str
        Formatted string like "0.850 [0.835, 0.865]"
    """
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(value)} [{fmt.format(ci_lower)}, {fmt.format(ci_upper)}]"
