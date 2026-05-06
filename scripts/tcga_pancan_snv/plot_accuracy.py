"""Generate the masked-token accuracy figures for the TCGA SNV pretraining (Figure 1 b-g, Supplementary Figures 1-2).

Reads per-variant loss + reconstruction predictions for all 7 trained
TESSERA SNV models from ``var_loss/`` (produced by
``get_variant_loss_acc.py``), joins them to the TCGA train/valid SNV
tables, computes per-sample / per-tumor-type / per-burden masked-token
accuracy, and writes the manuscript figures to ``plots/accuracy_analysis/``.

Usage
-----
    python plot_accuracy.py                                  # defaults
    USE_BOOTSTRAP=0 python plot_accuracy.py                  # skip 95% CI bootstrap (faster)
    LOSS_DIR=var_loss/ OUTPUT_DIR=plots/accuracy_analysis python plot_accuracy.py
"""

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import log_softmax

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
LOSS_FILES_DIR = os.environ.get("LOSS_DIR", "var_loss/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "plots/accuracy_analysis")
USE_BOOTSTRAP = os.environ.get("USE_BOOTSTRAP", "1") != "0"
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_snv.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_snv.csv")
MATPLOTLIB_BACKEND = os.environ.get("MATPLOTLIB_BACKEND")  # e.g. "Agg" for headless

if MATPLOTLIB_BACKEND:
    matplotlib.use(MATPLOTLIB_BACKEND, force=True)

# Reproducible bootstrap sampling.
np.random.seed(42)

n_bootstrap = N_BOOTSTRAP if USE_BOOTSTRAP else 0
plot_dir = OUTPUT_DIR
os.makedirs(plot_dir, exist_ok=True)

# Load train and valid data once
train_data = pd.read_csv(TRAIN_DATA_PATH)
valid_data = pd.read_csv(VALID_DATA_PATH)
combined_data = pd.concat([train_data, valid_data], ignore_index=True)

# Load all model results and compute accuracy
files = os.listdir(LOSS_FILES_DIR)
accuracy_data = []

for file in files:
    if file.endswith('.pkl'):
        label = '_'.join([file.split('_')[3], file.split('_')[4]])

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            out_train, out_valid = pickle.load(f)

        y_pred_train, logits_train, y_true_train, loss_train, y_pred_ref_train, logits_ref_train, y_true_ref_train = out_train
        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

        # Calculate accuracy for train
        train_pred_classes_alt = np.argmax(y_pred_train, axis=-1)
        train_pred_classes_ref = np.argmax(y_pred_ref_train, axis=-1)

        # Alt correct: all alt positions correct
        train_acc_alt = np.all(train_pred_classes_alt == y_true_train, axis=1).astype(int)
        # Ref correct: all ref positions correct
        train_acc_ref = np.all(train_pred_classes_ref == y_true_ref_train, axis=1).astype(int)
        # Combined: BOTH ref and alt correct
        train_acc_combined = (train_acc_alt & train_acc_ref).astype(int)

        # Calculate accuracy for valid
        valid_pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        valid_pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)

        # Alt correct: all alt positions correct
        valid_acc_alt = np.all(valid_pred_classes_alt == y_true_valid, axis=1).astype(int)
        # Ref correct: all ref positions correct
        valid_acc_ref = np.all(valid_pred_classes_ref == y_true_ref_valid, axis=1).astype(int)
        # Combined: BOTH ref and alt correct
        valid_acc_combined = (valid_acc_alt & valid_acc_ref).astype(int)

        # Store data for train
        for acc_val, acc_type in [(train_acc_combined, 'Combined'), (train_acc_ref, 'Ref'), (train_acc_alt, 'Alt')]:
            accuracy_data.append(pd.DataFrame({
                'Model': label,
                'Dataset': 'Train',
                'Accuracy': acc_val,
                'Type': acc_type
            }))

        # Store data for valid
        for acc_val, acc_type in [(valid_acc_combined, 'Combined'), (valid_acc_ref, 'Ref'), (valid_acc_alt, 'Alt')]:
            accuracy_data.append(pd.DataFrame({
                'Model': label,
                'Dataset': 'Valid',
                'Accuracy': acc_val,
                'Type': acc_type
            }))

# Convert to DataFrame
df = pd.concat(accuracy_data, ignore_index=True)

print("Accuracy data loaded successfully!")
print(f"Models found: {df['Model'].unique()}")
print(f"Datasets: {df['Dataset'].unique()}")
print(f"Types: {df['Type'].unique()}")
print(f"Shape: {df.shape}")

# ============================================================================
# MANUSCRIPT FIGURES - Accuracy Analysis
# ============================================================================
order = ['baseline_loss','local_1','global_1','local_10','global_10','local_25','global_25']

# Define colorblind-safe color palette (Paul Tol scheme - publication quality)
# Maintains semantic grouping: baseline gray, context families with light/dark pairs
# Passes deuteranopia, protanopia, and tritanopia tests
color_palette = {
    'baseline_loss': '#777777',  # Gray baseline
    'local_1':  '#88CCEE',       # Cyan (light)
    'global_1': '#0077BB',       # Blue (dark)
    'local_10':  '#DDAA33',      # Orange (light)
    'global_10': '#CC3311',      # Red (dark)
    'local_25':  '#AA3377',      # Magenta (light)
    'global_25': '#661100'       # Burgundy (dark)
}
colors = [color_palette[model] for model in order]

# Map data names to display names for figures
display_names = {
    'baseline_loss': 'baseline',
    'local_1': 'local_1',
    'global_1': 'global_1',
    'local_10': 'local_10',
    'global_10': 'global_10',
    'local_25': 'local_25',
    'global_25': 'global_25'
}

# Figure 1: Per-variant accuracy across models, stratified by type (Combined, Ref, Alt)
# Use validation set only for manuscript
df_valid = df[df['Dataset'] == 'Valid']

# Calculate mean accuracy and optionally bootstrap for confidence intervals
bootstrap_results = []

for model in order:
    for acc_type in ['Combined', 'Ref', 'Alt']:
        # Get accuracy values for this model and type
        mask = (df_valid['Model'] == model) & (df_valid['Type'] == acc_type)
        acc_values = df_valid[mask]['Accuracy'].values

        if len(acc_values) == 0:
            continue

        # Calculate mean accuracy
        mean_acc = np.mean(acc_values)

        if USE_BOOTSTRAP:
            # Bootstrap sampling for confidence intervals
            n_samples = len(acc_values)
            bootstrap_means = []

            for _ in range(n_bootstrap):
                sample = np.random.choice(acc_values, size=n_samples, replace=True)
                bootstrap_means.append(np.mean(sample))

            # Calculate 95% CI
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
        else:
            # No bootstrapping - no error bars
            ci_lower = mean_acc
            ci_upper = mean_acc

        bootstrap_results.append({
            'Model': model,
            'Type': acc_type,
            'Mean_Accuracy': mean_acc,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Error_Lower': mean_acc - ci_lower,
            'Error_Upper': ci_upper - mean_acc
        })

df_bootstrap = pd.DataFrame(bootstrap_results)

# Create publication-quality bar plot with error bars
fig, ax = plt.subplots(figsize=(10.0, 5.0))

# Set up positions for grouped bars
# Map data names to display names
type_data = ['Combined', 'Ref', 'Alt']  # Actual names in data
type_display = ['Ref/Alt', 'Ref', 'Alt']  # Display names for figure
x = np.arange(len(type_display))
width = 0.11  # Width for each bar
n_models = len(order)

for i, model in enumerate(order):
    model_data = df_bootstrap[df_bootstrap['Model'] == model]

    positions = x + (i - n_models/2 + 0.5) * width
    means = [model_data[model_data['Type'] == t]['Mean_Accuracy'].values[0]
             if len(model_data[model_data['Type'] == t]) > 0 else 0
             for t in type_data]
    errors_lower = [model_data[model_data['Type'] == t]['Error_Lower'].values[0]
                    if len(model_data[model_data['Type'] == t]) > 0 else 0
                    for t in type_data]
    errors_upper = [model_data[model_data['Type'] == t]['Error_Upper'].values[0]
                    if len(model_data[model_data['Type'] == t]) > 0 else 0
                    for t in type_data]

    if USE_BOOTSTRAP:
        bars = ax.bar(positions, means, width, label=display_names[model], color=colors[i], alpha=0.85,
                      yerr=[errors_lower, errors_upper], capsize=2.5, error_kw={'linewidth': 1.2})
    else:
        bars = ax.bar(positions, means, width, label=display_names[model], color=colors[i], alpha=0.85)

    # Add value labels on top of bars
    for bar, mean_val in zip(bars, means):
        if mean_val > 0:  # Only add label if value is not zero
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Styling and typography
ax.set_xlabel('Sequence Type', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Per-Variant Accuracy Across Models by Sequence Type',
             fontsize=24, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(type_display, fontsize=9)
ax.tick_params(axis='y', labelsize=9)
ax.set_ylim([0, 1.0])

# Add horizontal grid for readability
ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.25, color='gray')
ax.set_axisbelow(True)

# Legend inside plot
ax.legend(
    title='Model',
    fontsize=10,
    title_fontsize=11,
    loc='upper left',
    ncol=1,
    frameon=True,
    framealpha=0.95,
    edgecolor='gray'
)

# Remove top and right spines
sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'fig1_accuracy_by_model_and_type.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Figure 1: Per-Variant Accuracy (600 DPI PNG)")

# Compute pairwise statistics comparing models within each sequence type
from scipy import stats
pairwise_stats = []

for acc_type in type_data:  # Compare models within each sequence type
    # Get all pairwise comparisons between models
    for i, model1 in enumerate(order):
        for model2 in order[i+1:]:
            mask1 = (df_valid['Model'] == model1) & (df_valid['Type'] == acc_type)
            mask2 = (df_valid['Model'] == model2) & (df_valid['Type'] == acc_type)

            acc_values1 = df_valid[mask1]['Accuracy'].values
            acc_values2 = df_valid[mask2]['Accuracy'].values

            if len(acc_values1) > 0 and len(acc_values2) > 0:
                # Compute both t-test and Mann-Whitney U test
                t_stat, t_pvalue = stats.ttest_ind(acc_values1, acc_values2)
                u_stat, u_pvalue = stats.mannwhitneyu(acc_values1, acc_values2, alternative='two-sided')
                mean_diff = np.mean(acc_values1) - np.mean(acc_values2)
                median_diff = np.median(acc_values1) - np.median(acc_values2)

                pairwise_stats.append({
                    'Sequence_Type': acc_type,
                    'Model_1': model1,
                    'Model_2': model2,
                    'Mean_Acc_1': np.mean(acc_values1),
                    'Mean_Acc_2': np.mean(acc_values2),
                    'Median_Acc_1': np.median(acc_values1),
                    'Median_Acc_2': np.median(acc_values2),
                    'Mean_Difference': mean_diff,
                    'Median_Difference': median_diff,
                    'T_Statistic': t_stat,
                    'T_Test_P_Value': t_pvalue,
                    'Mann_Whitney_U_Statistic': u_stat,
                    'Mann_Whitney_U_P_Value': u_pvalue,
                    'N_1': len(acc_values1),
                    'N_2': len(acc_values2)
                })

# Save pairwise statistics to CSV
df_pairwise = pd.DataFrame(pairwise_stats)
# Apply display name mapping to model columns
df_pairwise['Model_1'] = df_pairwise['Model_1'].map(display_names)
df_pairwise['Model_2'] = df_pairwise['Model_2'].map(display_names)
stats_file = os.path.join(plot_dir, 'fig1_model_comparison_pairwise_stats.csv')
df_pairwise.to_csv(stats_file, index=False)
print(f"✓ Pairwise statistics saved to {stats_file}")

# ============================================================================
# Figure 2: Per-Sample Average Accuracy
# ============================================================================
# Need to add sample IDs to the accuracy data
sample_accuracy_data = []

for file in files:
    if file.endswith('.pkl'):
        label = '_'.join([file.split('_')[3], file.split('_')[4]])

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            out_train, out_valid = pickle.load(f)

        y_pred_train, logits_train, y_true_train, loss_train, y_pred_ref_train, logits_ref_train, y_true_ref_train = out_train
        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

        # Calculate accuracy for train
        train_pred_classes_alt = np.argmax(y_pred_train, axis=-1)
        train_pred_classes_ref = np.argmax(y_pred_ref_train, axis=-1)
        train_acc_alt = np.all(train_pred_classes_alt == y_true_train, axis=1).astype(int)
        train_acc_ref = np.all(train_pred_classes_ref == y_true_ref_train, axis=1).astype(int)
        train_acc_combined = (train_acc_alt & train_acc_ref).astype(int)

        # Calculate accuracy for valid
        valid_pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        valid_pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
        valid_acc_alt = np.all(valid_pred_classes_alt == y_true_valid, axis=1).astype(int)
        valid_acc_ref = np.all(valid_pred_classes_ref == y_true_ref_valid, axis=1).astype(int)
        valid_acc_combined = (valid_acc_alt & valid_acc_ref).astype(int)

        # Add samples from train data
        for i, sample in enumerate(train_data['Tumor_Sample_Barcode'].values):
            sample_accuracy_data.append({
                'Model': label,
                'Dataset': 'Train',
                'Sample': sample,
                'Acc_Combined': train_acc_combined[i],
                'Acc_Ref': train_acc_ref[i],
                'Acc_Alt': train_acc_alt[i]
            })

        # Add samples from valid data
        for i, sample in enumerate(valid_data['Tumor_Sample_Barcode'].values):
            sample_accuracy_data.append({
                'Model': label,
                'Dataset': 'Valid',
                'Sample': sample,
                'Acc_Combined': valid_acc_combined[i],
                'Acc_Ref': valid_acc_ref[i],
                'Acc_Alt': valid_acc_alt[i]
            })

# Convert to DataFrame
df_samples = pd.DataFrame(sample_accuracy_data)

# Calculate per-sample average accuracy for each model and type
df_sample_avg = df_samples.groupby(['Model', 'Dataset', 'Sample']).agg({
    'Acc_Combined': 'mean',
    'Acc_Ref': 'mean',
    'Acc_Alt': 'mean'
}).reset_index()

# Reshape for plotting
df_sample_avg_long = pd.melt(
    df_sample_avg,
    id_vars=['Model', 'Dataset', 'Sample'],
    value_vars=['Acc_Combined', 'Acc_Ref', 'Acc_Alt'],
    var_name='Type',
    value_name='Avg_Accuracy'
)
# Clean up type names
df_sample_avg_long['Type'] = df_sample_avg_long['Type'].str.replace('Acc_', '')

# Filter to validation set only
df_sample_avg_valid = df_sample_avg_long[df_sample_avg_long['Dataset'] == 'Valid']

# Plot - Publication quality
fig, ax = plt.subplots(figsize=(7.0, 4.5))
sns.boxplot(data=df_sample_avg_valid, hue='Model', y='Avg_Accuracy', x='Type',
               hue_order=order, order=['Combined', 'Ref', 'Alt'],
               palette=colors, showfliers=False, ax=ax, width=0.7,
               linewidth=1.5, whis=0.75)  # Reduced whisker interval from default 1.5 to 0.75

ax.set_title('Per-Sample Average Accuracy Distribution Across Models',
             fontsize=24, fontweight='bold', pad=15)
ax.set_ylabel('Average Accuracy per Sample', fontsize=11)
ax.set_xlabel('Sequence Type', fontsize=11)
ax.tick_params(axis='both', labelsize=9)

# Update x-axis labels: Combined -> Ref/Alt
ax.set_xticklabels(['Ref/Alt', 'Ref', 'Alt'])

# Add horizontal grid
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legend inside plot - update labels to use display names
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [display_names.get(label, label) for label in labels],
          title='Model', fontsize=9, title_fontsize=10,
          loc='lower right', ncol=2, frameon=True, framealpha=0.95,
          edgecolor='gray', columnspacing=1.0, handlelength=1.5)

# Remove top and right spines
sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'fig2_per_sample_avg_accuracy.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Figure 2: Per-Sample Average Accuracy (600 DPI PNG)")

# Compute pairwise statistics for Figure 2 (per-sample average accuracy)
pairwise_stats_fig2 = []

for acc_type in type_data:  # Compare models within each sequence type (Combined, Ref, Alt)
    # Get all pairwise comparisons between models
    for i, model1 in enumerate(order):
        for model2 in order[i+1:]:
            mask1 = (df_sample_avg_valid['Model'] == model1) & (df_sample_avg_valid['Type'] == acc_type)
            mask2 = (df_sample_avg_valid['Model'] == model2) & (df_sample_avg_valid['Type'] == acc_type)

            acc_values1 = df_sample_avg_valid[mask1]['Avg_Accuracy'].values
            acc_values2 = df_sample_avg_valid[mask2]['Avg_Accuracy'].values

            if len(acc_values1) > 0 and len(acc_values2) > 0:
                # Compute both t-test and Mann-Whitney U test
                t_stat, t_pvalue = stats.ttest_ind(acc_values1, acc_values2)
                u_stat, u_pvalue = stats.mannwhitneyu(acc_values1, acc_values2, alternative='two-sided')
                mean_diff = np.mean(acc_values1) - np.mean(acc_values2)
                median_diff = np.median(acc_values1) - np.median(acc_values2)

                pairwise_stats_fig2.append({
                    'Sequence_Type': acc_type,
                    'Model_1': model1,
                    'Model_2': model2,
                    'Mean_Acc_1': np.mean(acc_values1),
                    'Mean_Acc_2': np.mean(acc_values2),
                    'Median_Acc_1': np.median(acc_values1),
                    'Median_Acc_2': np.median(acc_values2),
                    'Mean_Difference': mean_diff,
                    'Median_Difference': median_diff,
                    'T_Statistic': t_stat,
                    'T_Test_P_Value': t_pvalue,
                    'Mann_Whitney_U_Statistic': u_stat,
                    'Mann_Whitney_U_P_Value': u_pvalue,
                    'N_1': len(acc_values1),
                    'N_2': len(acc_values2)
                })

# Save pairwise statistics to CSV
df_pairwise_fig2 = pd.DataFrame(pairwise_stats_fig2)
# Apply display name mapping to model columns
df_pairwise_fig2['Model_1'] = df_pairwise_fig2['Model_1'].map(display_names)
df_pairwise_fig2['Model_2'] = df_pairwise_fig2['Model_2'].map(display_names)
stats_file_fig2 = os.path.join(plot_dir, 'fig2_model_comparison_pairwise_stats.csv')
df_pairwise_fig2.to_csv(stats_file_fig2, index=False)
print(f"✓ Pairwise statistics saved to {stats_file_fig2}")

# ============================================================================
# Figure 3: Per-Variant Accuracy by Number of Variants per Sample (Bar Plot)
# ============================================================================

# Choose binning method: 'quantile' or 'fixed'
binning_method = 'fixed'  # Change to 'fixed' for even number ranges

# Set up bin edges and labels for fixed binning
if binning_method == 'fixed':
    bin_edges = [1, 200, 400, 600, 800, 1000, 1200, np.inf]
    bin_labels = ['1-200', '200-400', '400-600', '600-800', '800-1000', '1000+', '1200+']
    ordered_bins = bin_labels

# Count variants per sample from the original data
variant_counts_per_sample = combined_data.groupby('Tumor_Sample_Barcode').size().reset_index(name='Variant_Count')

# Prepare data: add variant count bins to the per-variant accuracy data
per_variant_data_with_bins = []

for file in files:
    if file.endswith('.pkl'):
        label = '_'.join([file.split('_')[3], file.split('_')[4]])

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            out_train, out_valid = pickle.load(f)

        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

        # Calculate combined accuracy (both alt and ref correct)
        pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        acc_alt = np.all(pred_classes_alt == y_true_valid, axis=1).astype(int)

        pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
        acc_ref = np.all(pred_classes_ref == y_true_ref_valid, axis=1).astype(int)

        acc_combined = (acc_alt & acc_ref).astype(int)

        # Vectorized: Create temporary dataframe and merge with variant counts
        temp_df = pd.DataFrame({
            'Tumor_Sample_Barcode': valid_data['Tumor_Sample_Barcode'].values,
            'Model': label,
            'Accuracy': acc_combined
        })

        # Merge with variant counts
        temp_df = temp_df.merge(variant_counts_per_sample, on='Tumor_Sample_Barcode', how='left')

        # Assign to bin based on current binning method
        if binning_method == 'fixed':
            # Use pandas cut (vectorized)
            temp_df['Bin'] = pd.cut(
                temp_df['Variant_Count'],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True,
                right=False
            )
        else:
            # For quantile method, merge with df_sample_variant_counts to get bins
            temp_df = temp_df.merge(
                df_sample_variant_counts[['Sample', 'Bin_Range']].rename(columns={'Sample': 'Tumor_Sample_Barcode'}),
                on='Tumor_Sample_Barcode',
                how='left'
            )
            temp_df['Bin'] = temp_df['Bin_Range']

        # Drop rows with missing bins and append
        temp_df = temp_df.dropna(subset=['Bin'])
        per_variant_data_with_bins.append(temp_df[['Model', 'Accuracy', 'Bin']])

# Convert to DataFrame (concatenate all model dataframes)
df_per_variant_bins = pd.concat(per_variant_data_with_bins, ignore_index=True)

# Filter to bins with data
bins_with_data = [b for b in ordered_bins if b in df_per_variant_bins['Bin'].unique()]

print(f"\nPer-variant accuracy by bin - Bins to plot: {bins_with_data}")

# Calculate mean accuracy and optionally bootstrap for confidence intervals
bootstrap_results_bins = []

for model in order:
    for bin_label in bins_with_data:
        # Get accuracy values for this model and bin
        mask = (df_per_variant_bins['Model'] == model) & (df_per_variant_bins['Bin'] == bin_label)
        acc_values = df_per_variant_bins[mask]['Accuracy'].values

        if len(acc_values) == 0:
            continue

        # Calculate mean accuracy
        mean_acc = np.mean(acc_values)

        if USE_BOOTSTRAP:
            # Bootstrap sampling for confidence intervals
            n_samples = len(acc_values)
            bootstrap_means = []

            for _ in range(n_bootstrap):
                sample = np.random.choice(acc_values, size=n_samples, replace=True)
                bootstrap_means.append(np.mean(sample))

            # Calculate 95% CI
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
        else:
            # No bootstrapping - no error bars
            ci_lower = mean_acc
            ci_upper = mean_acc

        bootstrap_results_bins.append({
            'Model': model,
            'Bin': bin_label,
            'Mean_Accuracy': mean_acc,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Error_Lower': mean_acc - ci_lower,
            'Error_Upper': ci_upper - mean_acc
        })

df_bootstrap_bins = pd.DataFrame(bootstrap_results_bins)

# Create publication-quality grouped bar plot
fig, ax = plt.subplots(figsize=(10.0, 5.0))

x = np.arange(len(bins_with_data))
width = 0.11  # Narrower bars for better appearance
n_models = len(order)

for i, model in enumerate(order):
    model_data = df_bootstrap_bins[df_bootstrap_bins['Model'] == model]

    positions = x + (i - n_models/2 + 0.5) * width
    means = [model_data[model_data['Bin'] == b]['Mean_Accuracy'].values[0]
             if len(model_data[model_data['Bin'] == b]) > 0 else 0
             for b in bins_with_data]
    errors_lower = [model_data[model_data['Bin'] == b]['Error_Lower'].values[0]
                    if len(model_data[model_data['Bin'] == b]) > 0 else 0
                    for b in bins_with_data]
    errors_upper = [model_data[model_data['Bin'] == b]['Error_Upper'].values[0]
                    if len(model_data[model_data['Bin'] == b]) > 0 else 0
                    for b in bins_with_data]

    if USE_BOOTSTRAP:
        bars = ax.bar(positions, means, width, label=model,
                      color=color_palette.get(model, '#808080'), alpha=0.85,
                      yerr=[errors_lower, errors_upper], capsize=2, error_kw={'linewidth': 1, 'alpha': 0.7})
    else:
        bars = ax.bar(positions, means, width, label=model,
                      color=color_palette.get(model, '#808080'), alpha=0.85)

    # Add value labels on top of bars in black
    for bar, mean_val in zip(bars, means):
        if mean_val > 0:  # Only add label if value is not zero
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=4, color='black', weight='bold')

ax.set_xlabel('Number of Variants per Sample', fontsize=11)
ax.set_ylabel('Per-Variant Accuracy', fontsize=11)
ax.set_title(f'Per-Variant Accuracy by Variant Burden',
             fontsize=24, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(bins_with_data, rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelsize=9)

# Add horizontal grid
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

ax.set_ylim([0, 1])
sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'fig3_per_variant_accuracy_by_variant_count.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Figure 3: Per-Variant Accuracy by Variant Burden (600 DPI PNG)")

# ============================================================================
# Figure 4: Per-Sample Average Accuracy by Number of Variants per Sample
# ============================================================================
# Load sample accuracy data independently for this figure
sample_accuracy_data_fig3 = []

for file in files:
    if file.endswith('.pkl'):
        label = '_'.join([file.split('_')[3], file.split('_')[4]])

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            out_train, out_valid = pickle.load(f)

        y_pred_train, logits_train, y_true_train, loss_train, y_pred_ref_train, logits_ref_train, y_true_ref_train = out_train
        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

        # Calculate accuracy for valid
        valid_pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        valid_pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
        valid_acc_alt = np.all(valid_pred_classes_alt == y_true_valid, axis=1).astype(int)
        valid_acc_ref = np.all(valid_pred_classes_ref == y_true_ref_valid, axis=1).astype(int)
        valid_acc_combined = (valid_acc_alt & valid_acc_ref).astype(int)

        # Add samples from valid data
        for i, sample in enumerate(valid_data['Tumor_Sample_Barcode'].values):
            sample_accuracy_data_fig3.append({
                'Model': label,
                'Sample': sample,
                'Acc_Combined': valid_acc_combined[i]
            })

# Convert to DataFrame
df_samples_fig3 = pd.DataFrame(sample_accuracy_data_fig3)

# Calculate per-sample average accuracy for each model
df_sample_avg_fig3 = df_samples_fig3.groupby(['Model', 'Sample']).agg({
    'Acc_Combined': 'mean'
}).reset_index()
df_sample_avg_fig3.columns = ['Model', 'Sample', 'Avg_Accuracy']

# Count variants per sample from the original data
variant_counts_per_sample = combined_data.groupby('Tumor_Sample_Barcode').size().reset_index(name='Variant_Count')

# Add variant counts to the sample average dataframe
df_sample_variant_counts = df_sample_avg_fig3.merge(
    variant_counts_per_sample,
    left_on='Sample',
    right_on='Tumor_Sample_Barcode',
    how='left'
)

# binning_method already defined in Figure 3, reusing the same binning approach
if binning_method == 'quantile':
    # Create quantile-based bins (5 bins with equal sample counts)
    n_bins = 5
    try:
        df_sample_variant_counts['Variant_Bin'] = pd.qcut(
            df_sample_variant_counts['Variant_Count'],
            q=n_bins,
            labels=False,
            duplicates='drop'
        )
    except ValueError:
        # If qcut fails due to duplicates, use rank-based approach
        df_sample_variant_counts['Variant_Bin'] = pd.qcut(
            df_sample_variant_counts['Variant_Count'].rank(method='first'),
            q=n_bins,
            labels=False
        )

    # Get bin ranges for labeling
    bin_ranges = []
    for bin_num in sorted(df_sample_variant_counts['Variant_Bin'].unique()):
        bin_data = df_sample_variant_counts[df_sample_variant_counts['Variant_Bin'] == bin_num]
        min_count = int(bin_data['Variant_Count'].min())
        max_count = int(bin_data['Variant_Count'].max())
        bin_ranges.append((bin_num, f"{min_count}-{max_count}"))

    bin_range_map = dict(bin_ranges)
    df_sample_variant_counts['Bin_Range'] = df_sample_variant_counts['Variant_Bin'].map(bin_range_map)
    title_suffix = '(Quantile Bins)'

elif binning_method == 'fixed':
    # Create fixed range bins (1-200, 200-400, etc., 1000-1200, 1200+)
    bin_edges = [1, 200, 400, 600, 800, 1000, 1200, np.inf]
    bin_labels = ['1-200', '200-400', '400-600', '600-800', '800-1000', '1000+', '1200+']

    df_sample_variant_counts['Variant_Bin'] = pd.cut(
        df_sample_variant_counts['Variant_Count'],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False  # Makes bins like [1, 200), [200, 400), etc.
    )

    df_sample_variant_counts['Bin_Range'] = df_sample_variant_counts['Variant_Bin']

    # Only keep bins that have data
    bin_counts = df_sample_variant_counts['Bin_Range'].value_counts()
    ordered_bins = [label for label in bin_labels if label in bin_counts.index and bin_counts[label] > 0]
    title_suffix = '(Fixed Range Bins)'

# Create publication-quality boxplot
if binning_method == 'quantile':
    ordered_bins = [bin_range_map[i] for i in sorted(bin_range_map.keys())]

fig, ax = plt.subplots(figsize=(7.0, 4.5))
sns.boxplot(data=df_sample_variant_counts, x='Bin_Range', y='Avg_Accuracy', hue='Model',
               order=ordered_bins, hue_order=order, palette=colors, showfliers=False,
               ax=ax, width=0.7, linewidth=1.5, whis=0.75)

ax.set_title(f'Per-Sample Average Accuracy by Variant Burden',
             fontsize=24, fontweight='bold', pad=15)
ax.set_ylabel('Average Accuracy per Sample', fontsize=11)
ax.set_xlabel('Number of Variants per Sample', fontsize=11)
ax.tick_params(axis='x', labelsize=8, rotation=45)
ax.tick_params(axis='y', labelsize=9)

# Add horizontal grid
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
ax.set_axisbelow(True)

# Legend inside plot - update labels to use display names
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [display_names.get(label, label) for label in labels],
          title='Model', fontsize=8, title_fontsize=9,
          loc='upper left', ncol=2, frameon=True, framealpha=0.95,
          edgecolor='gray', columnspacing=0.8, handlelength=1.2)

sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'fig4_per_sample_accuracy_by_variant_count.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Figure 4: Per-Sample Accuracy by Variant Burden (600 DPI PNG)")

# ============================================================================
# Figure 5: Per-Sample Average Accuracy by Tumor Type
# ============================================================================
# Load sample accuracy data independently for this figure
sample_accuracy_data_fig4 = []

for file in files:
    if file.endswith('.pkl'):
        label = '_'.join([file.split('_')[3], file.split('_')[4]])

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            out_train, out_valid = pickle.load(f)

        y_pred_train, logits_train, y_true_train, loss_train, y_pred_ref_train, logits_ref_train, y_true_ref_train = out_train
        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

        # Calculate accuracy for valid
        valid_pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        valid_pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
        valid_acc_alt = np.all(valid_pred_classes_alt == y_true_valid, axis=1).astype(int)
        valid_acc_ref = np.all(valid_pred_classes_ref == y_true_ref_valid, axis=1).astype(int)
        valid_acc_combined = (valid_acc_alt & valid_acc_ref).astype(int)

        # Add samples from valid data
        for i, sample in enumerate(valid_data['Tumor_Sample_Barcode'].values):
            sample_accuracy_data_fig4.append({
                'Model': label,
                'Sample': sample,
                'Acc_Combined': valid_acc_combined[i]
            })

# Convert to DataFrame
df_samples_fig4 = pd.DataFrame(sample_accuracy_data_fig4)

# Calculate per-sample average accuracy for each model
df_sample_avg_fig4 = df_samples_fig4.groupby(['Model', 'Sample']).agg({
    'Acc_Combined': 'mean'
}).reset_index()
df_sample_avg_fig4.columns = ['Model', 'Sample', 'Avg_Accuracy']

# Get tumor type information from the 'type' column
if 'type' in combined_data.columns:
    sample_tumor_types = combined_data.groupby('Tumor_Sample_Barcode')['type'].first().reset_index()
    sample_tumor_types.columns = ['Tumor_Sample_Barcode', 'tumor_type']
else:
    print("Warning: Could not find 'type' column. Skipping Figure 5.")
    sample_tumor_types = None

if sample_tumor_types is not None:
    # Merge tumor types with sample accuracy data
    df_sample_tumor = df_sample_avg_fig4.merge(
        sample_tumor_types,
        left_on='Sample',
        right_on='Tumor_Sample_Barcode',
        how='left'
    )

    # Get all tumor types sorted by sample count
    tumor_type_counts = df_sample_tumor.groupby('tumor_type').size().sort_values(ascending=False)
    all_tumor_types = tumor_type_counts.index.tolist()

    print(f"All tumor types ({len(all_tumor_types)}): {all_tumor_types}")

    # Create subfolder for tumor type plots
    tumor_type_plot_dir = os.path.join(plot_dir, 'fig5_tumor_types')
    if not os.path.exists(tumor_type_plot_dir):
        os.makedirs(tumor_type_plot_dir)

    # Create individual boxplot for each tumor type (publication quality)
    for tumor_type in all_tumor_types:
        # Filter data for this tumor type
        df_tumor_type = df_sample_tumor[df_sample_tumor['tumor_type'] == tumor_type]
        n_samples = len(df_tumor_type) // len(order)  # Approx samples per model

        # Create publication-quality boxplot
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        sns.boxplot(data=df_tumor_type, x='Model', y='Avg_Accuracy',
                    order=order, palette=colors, showfliers=False, ax=ax,
                    width=0.6, linewidth=1.2, whis=0.75)

        ax.set_title(f'{tumor_type} (n={n_samples})',
                     fontsize=11, fontweight='bold', pad=12)
        ax.set_ylabel('Average Accuracy per Sample', fontsize=10)
        ax.set_xlabel('Model', fontsize=10)
        ax.tick_params(axis='x', labelsize=8, rotation=45)

        # Update x-axis labels to use display names
        ax.set_xticklabels([display_names.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()])
        ax.tick_params(axis='y', labelsize=8)
        ax.set_ylim([0, 1])

        # Add grid
        ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.2, color='gray')
        ax.set_axisbelow(True)

        sns.despine()
        plt.tight_layout()

        # Save PNG only
        plt.savefig(os.path.join(tumor_type_plot_dir, f'{tumor_type}_accuracy.png'), dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"Figure 5: Saved {len(all_tumor_types)} tumor type plots (600 DPI PNG)")

    # ============================================================================
    # Figure 6: Per-Variant Accuracy by Tumor Type (Bar Plot)
    # ============================================================================

    # Prepare data: add tumor type to per-variant accuracy data
    per_variant_data_with_tumor = []

    for file in files:
        if file.endswith('.pkl'):
            label = '_'.join([file.split('_')[3], file.split('_')[4]])

            with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
                out_train, out_valid = pickle.load(f)

            y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

            # Calculate combined accuracy (both alt and ref correct)
            pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
            acc_alt = np.all(pred_classes_alt == y_true_valid, axis=1).astype(int)

            pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
            acc_ref = np.all(pred_classes_ref == y_true_ref_valid, axis=1).astype(int)

            acc_combined = (acc_alt & acc_ref).astype(int)

            # Vectorized: Create temporary dataframe and merge with tumor types
            temp_df = pd.DataFrame({
                'Tumor_Sample_Barcode': valid_data['Tumor_Sample_Barcode'].values,
                'Model': label,
                'Accuracy': acc_combined
            })

            # Merge with tumor types
            temp_df = temp_df.merge(sample_tumor_types, on='Tumor_Sample_Barcode', how='left')

            # Drop rows with missing tumor types and append
            temp_df = temp_df.dropna(subset=['tumor_type'])
            per_variant_data_with_tumor.append(temp_df[['Model', 'Accuracy', 'tumor_type']].rename(columns={'tumor_type': 'Tumor_Type'}))

    # Convert to DataFrame (concatenate all model dataframes)
    df_per_variant_tumor = pd.concat(per_variant_data_with_tumor, ignore_index=True)

    # Sort tumor types by per-variant mean accuracy (across all models and variants)
    tumor_type_per_variant_mean = df_per_variant_tumor.groupby('Tumor_Type')['Accuracy'].mean().sort_values(ascending=False)
    tumor_types_to_plot = tumor_type_per_variant_mean.index.tolist()

    print(f"\nPer-variant accuracy by tumor type - Types to plot: {len(tumor_types_to_plot)}")

    # Calculate mean accuracy and optionally bootstrap for confidence intervals
    bootstrap_results_tumor = []

    for model in order:
        for tumor_type in tumor_types_to_plot:
            # Get accuracy values for this model and tumor type
            mask = (df_per_variant_tumor['Model'] == model) & (df_per_variant_tumor['Tumor_Type'] == tumor_type)
            acc_values = df_per_variant_tumor[mask]['Accuracy'].values

            if len(acc_values) == 0:
                continue

            # Calculate mean accuracy
            mean_acc = np.mean(acc_values)

            if USE_BOOTSTRAP:
                # Bootstrap sampling for confidence intervals
                n_samples = len(acc_values)
                bootstrap_means = []

                for _ in range(n_bootstrap):
                    sample = np.random.choice(acc_values, size=n_samples, replace=True)
                    bootstrap_means.append(np.mean(sample))

                # Calculate 95% CI
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
            else:
                # No bootstrapping - no error bars
                ci_lower = mean_acc
                ci_upper = mean_acc

            bootstrap_results_tumor.append({
                'Model': model,
                'Tumor_Type': tumor_type,
                'Mean_Accuracy': mean_acc,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Error_Lower': mean_acc - ci_lower,
                'Error_Upper': ci_upper - mean_acc
            })

    df_bootstrap_tumor = pd.DataFrame(bootstrap_results_tumor)

    # Create publication-quality grouped bar plot
    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    x = np.arange(len(tumor_types_to_plot))
    width = 0.10  # Narrower bars
    n_models = len(order)

    for i, model in enumerate(order):
        model_data = df_bootstrap_tumor[df_bootstrap_tumor['Model'] == model]

        positions = x + (i - n_models/2 + 0.5) * width
        means = [model_data[model_data['Tumor_Type'] == tt]['Mean_Accuracy'].values[0]
                 if len(model_data[model_data['Tumor_Type'] == tt]) > 0 else 0
                 for tt in tumor_types_to_plot]
        errors_lower = [model_data[model_data['Tumor_Type'] == tt]['Error_Lower'].values[0]
                        if len(model_data[model_data['Tumor_Type'] == tt]) > 0 else 0
                        for tt in tumor_types_to_plot]
        errors_upper = [model_data[model_data['Tumor_Type'] == tt]['Error_Upper'].values[0]
                        if len(model_data[model_data['Tumor_Type'] == tt]) > 0 else 0
                        for tt in tumor_types_to_plot]

        if USE_BOOTSTRAP:
            bars = ax.bar(positions, means, width, label=model,
                          color=color_palette.get(model, '#808080'), alpha=0.85,
                          yerr=[errors_lower, errors_upper], capsize=1.5, error_kw={'linewidth': 0.8, 'alpha': 0.7})
        else:
            bars = ax.bar(positions, means, width, label=model,
                          color=color_palette.get(model, '#808080'), alpha=0.85)

    ax.set_xlabel('Tumor Type', fontsize=11)
    ax.set_ylabel('Per-Variant Accuracy', fontsize=11)
    ax.set_title('Per-Variant Accuracy by Tumor Type',
                 fontsize=24, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tumor_types_to_plot, rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    ax.set_ylim([0, 1])
    sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'fig6_per_variant_accuracy_by_tumor_type.png'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print("✓ Figure 6: Per-Variant Accuracy by Tumor Type (600 DPI PNG)")

# ============================================================================
# Figure 7: Mutation Count × Tumor Type Heatmap (Single Model)
# ============================================================================
print("\nGenerating Figure 7: Mutation Count × Tumor Type Heatmap")

# Select model for heatmap
heatmap_model = 'global_25'

# Load sample accuracy data for heatmap
sample_accuracy_data_fig9 = []

for file in files:
    if file.endswith('.pkl'):
        label = '_'.join([file.split('_')[3], file.split('_')[4]])

        # Only process the selected model
        if label != heatmap_model:
            continue

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            out_train, out_valid = pickle.load(f)

        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

        # Calculate accuracy for valid
        valid_pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        valid_pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
        valid_acc_alt = np.all(valid_pred_classes_alt == y_true_valid, axis=1).astype(int)
        valid_acc_ref = np.all(valid_pred_classes_ref == y_true_ref_valid, axis=1).astype(int)
        valid_acc_combined = (valid_acc_alt & valid_acc_ref).astype(int)

        # Add samples from valid data
        for i, sample in enumerate(valid_data['Tumor_Sample_Barcode'].values):
            sample_accuracy_data_fig9.append({
                'Model': label,
                'Sample': sample,
                'Acc_Combined': valid_acc_combined[i]
            })

# Convert to DataFrame
df_samples_fig9 = pd.DataFrame(sample_accuracy_data_fig9)

# Calculate per-sample average accuracy
df_sample_avg_fig9 = df_samples_fig9.groupby(['Model', 'Sample']).agg({
    'Acc_Combined': 'mean'
}).reset_index()
df_sample_avg_fig9.columns = ['Model', 'Sample', 'Avg_Accuracy']

# Merge with variant counts
df_heatmap = df_sample_avg_fig9.merge(
    variant_counts_per_sample,
    left_on='Sample',
    right_on='Tumor_Sample_Barcode',
    how='left'
)

# Apply mutation count binning (using existing bin_edges from Figure 3)
df_heatmap['Variant_Bin'] = pd.cut(
    df_heatmap['Variant_Count'],
    bins=bin_edges,
    labels=bin_labels,
    include_lowest=True,
    right=False
)

# Merge with tumor types
if sample_tumor_types is not None:
    df_heatmap = df_heatmap.merge(
        sample_tumor_types,
        on='Tumor_Sample_Barcode',
        how='left'
    )

    # Drop rows with missing data
    df_heatmap = df_heatmap.dropna(subset=['Variant_Bin', 'tumor_type'])

    # Create pivot table: rows = tumor types, columns = mutation bins, values = mean accuracy
    heatmap_pivot = df_heatmap.pivot_table(
        index='tumor_type',
        columns='Variant_Bin',
        values='Avg_Accuracy',
        aggfunc='mean'
    )

    # Create sample count pivot table
    sample_count_pivot = df_heatmap.pivot_table(
        index='tumor_type',
        columns='Variant_Bin',
        values='Sample',
        aggfunc='count'
    )

    # Sort tumor types by overall mean accuracy (descending)
    tumor_type_overall_acc = df_heatmap.groupby('tumor_type')['Avg_Accuracy'].mean().sort_values(ascending=False)
    heatmap_pivot = heatmap_pivot.reindex(tumor_type_overall_acc.index)
    sample_count_pivot = sample_count_pivot.reindex(tumor_type_overall_acc.index)

    # Only keep columns (bins) that have data
    heatmap_pivot = heatmap_pivot[ordered_bins]
    sample_count_pivot = sample_count_pivot[ordered_bins]

    # Create annotations with accuracy and sample count
    annot_labels = heatmap_pivot.copy()
    for i in range(len(annot_labels.index)):
        for j in range(len(annot_labels.columns)):
            acc_val = heatmap_pivot.iloc[i, j]
            count_val = sample_count_pivot.iloc[i, j]
            if pd.notna(acc_val) and pd.notna(count_val):
                annot_labels.iloc[i, j] = f'{acc_val:.2f} ({int(count_val)})'
            else:
                annot_labels.iloc[i, j] = ''

    print(f"Heatmap shape: {heatmap_pivot.shape}")
    print(f"Tumor types: {len(heatmap_pivot.index)}")
    print(f"Mutation bins: {len(heatmap_pivot.columns)}")

    # Create publication-quality heatmap
    fig, ax = plt.subplots(figsize=(7.0, 8.0))

    sns.heatmap(
        heatmap_pivot,
        annot=annot_labels,
        fmt='',
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Mean Accuracy', 'shrink': 0.8},
        linewidths=0.3,
        linecolor='gray',
        ax=ax,
        annot_kws={'fontsize': 7}
    )

    ax.set_title(f'Per-Sample Accuracy:\nMutation Count × Tumor Type ({heatmap_model})',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Mutation Count Bin', fontsize=10)
    ax.set_ylabel('Tumor Type', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f'fig7_accuracy_heatmap_mutation_x_tumor_type_{heatmap_model}.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Figure 7: Mutation Count × Tumor Type Heatmap (600 DPI PNG)")
else:
    print("Warning: Could not create Figure 7 - tumor type data not available")

# ============================================================================
# Figure 8: Absolute Improvement Heatmap (global_25 - local_25)
# ============================================================================
print("\nGenerating Figure 8: Absolute Improvement Heatmap (global_25 - local_25)")

# Prepare comparison data for Figure 8
# Load sample accuracy data for BOTH global_25 and local_25 models
comparison_data = []

for file in files:
    if file.endswith('.pkl'):
        label = '_'.join([file.split('_')[3], file.split('_')[4]])

        # Only process global_25 and local_25
        if label not in ['global_25', 'local_25']:
            continue

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            out_train, out_valid = pickle.load(f)

        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = out_valid

        # Calculate accuracy for valid
        valid_pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        valid_pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
        valid_acc_alt = np.all(valid_pred_classes_alt == y_true_valid, axis=1).astype(int)
        valid_acc_ref = np.all(valid_pred_classes_ref == y_true_ref_valid, axis=1).astype(int)
        valid_acc_combined = (valid_acc_alt & valid_acc_ref).astype(int)

        # Add samples from valid data
        for i, sample in enumerate(valid_data['Tumor_Sample_Barcode'].values):
            comparison_data.append({
                'Model': label,
                'Sample': sample,
                'Acc_Combined': valid_acc_combined[i]
            })

# Convert to DataFrame
df_comparison_raw = pd.DataFrame(comparison_data)

# Calculate per-sample average accuracy
df_comparison_avg = df_comparison_raw.groupby(['Model', 'Sample']).agg({
    'Acc_Combined': 'mean'
}).reset_index()
df_comparison_avg.columns = ['Model', 'Sample', 'Avg_Accuracy']

# Merge with variant counts
df_comparison = df_comparison_avg.merge(
    variant_counts_per_sample,
    left_on='Sample',
    right_on='Tumor_Sample_Barcode',
    how='left'
)

# Apply mutation count binning (reuse bin_edges from Figure 3)
df_comparison['Variant_Bin'] = pd.cut(
    df_comparison['Variant_Count'],
    bins=bin_edges,
    labels=bin_labels,
    include_lowest=True,
    right=False
)

# Merge with tumor types (if available)
if sample_tumor_types is not None:
    df_comparison = df_comparison.merge(
        sample_tumor_types,
        on='Tumor_Sample_Barcode',
        how='left'
    )
    # Rename type column first, then drop NAs
    df_comparison.rename(columns={'type': 'tumor_type'}, inplace=True)
    df_comparison = df_comparison.dropna(subset=['Variant_Bin', 'tumor_type'])

    # Create pivot tables for each model
    global_25_data = df_comparison[df_comparison['Model'] == 'global_25']
    local_25_data = df_comparison[df_comparison['Model'] == 'local_25']

    global_25_pivot = global_25_data.pivot_table(
        index='tumor_type',
        columns='Variant_Bin',
        values='Avg_Accuracy',
        aggfunc='mean'
    )

    local_25_pivot = local_25_data.pivot_table(
        index='tumor_type',
        columns='Variant_Bin',
        values='Avg_Accuracy',
        aggfunc='mean'
    )

    print(f"✓ Figure 8 data prepared: global_25_pivot shape {global_25_pivot.shape}, local_25_pivot shape {local_25_pivot.shape}")

    # Compute absolute improvement: global_25 - local_25
    absolute_improvement = global_25_pivot - local_25_pivot
else:
    print("Warning: Could not create Figure 8 - tumor type data not available")
    absolute_improvement = None

# Only generate Figure 8 if data is available
if absolute_improvement is not None:
    # Get sample counts (use global_25 data for counts)
    sample_count_pivot_fig11 = df_comparison[df_comparison['Model'] == 'global_25'].pivot_table(
        index='tumor_type',
        columns='Variant_Bin',
        values='Sample',
        aggfunc='count'
    )

    # Sort tumor types by overall mean improvement (descending)
    tumor_type_mean_abs_improvement = absolute_improvement.mean(axis=1).sort_values(ascending=False)
    absolute_improvement = absolute_improvement.reindex(tumor_type_mean_abs_improvement.index)
    sample_count_pivot_fig11 = sample_count_pivot_fig11.reindex(tumor_type_mean_abs_improvement.index)

    # Only keep columns (bins) that have data
    absolute_improvement = absolute_improvement[ordered_bins]
    sample_count_pivot_fig11 = sample_count_pivot_fig11[ordered_bins]

    # Create annotations with absolute improvement and sample count
    annot_labels_fig11 = absolute_improvement.copy()
    for i in range(len(annot_labels_fig11.index)):
        for j in range(len(annot_labels_fig11.columns)):
            abs_val = absolute_improvement.iloc[i, j]
            count_val = sample_count_pivot_fig11.iloc[i, j]
            if pd.notna(abs_val) and pd.notna(count_val):
                annot_labels_fig11.iloc[i, j] = f'{abs_val:.3f} ({int(count_val)})'
            else:
                annot_labels_fig11.iloc[i, j] = ''

    print(f"Absolute improvement heatmap shape: {absolute_improvement.shape}")
    print(f"Tumor types: {len(absolute_improvement.index)}")
    print(f"Mutation bins: {len(absolute_improvement.columns)}")

    # Create publication-quality heatmap
    fig, ax = plt.subplots(figsize=(7.0, 8.0))

    # Calculate actual min/max values
    abs_min = absolute_improvement.min().min()
    abs_max = absolute_improvement.max().max()

    print(f"Absolute improvement range: [{abs_min:.4f}, {abs_max:.4f}]")

    # Use diverging colormap centered at 0.0 with actual data range
    sns.heatmap(
        absolute_improvement,
        annot=annot_labels_fig11,
        fmt='',
        cmap='coolwarm',
        center=0.0,
        vmin=abs_min,
        vmax=abs_max,
        cbar_kws={'label': 'Absolute Improvement\n(global_25 - local_25)', 'shrink': 0.8},
        linewidths=0.3,
        linecolor='gray',
        ax=ax,
        annot_kws={'fontsize': 7}
    )

    ax.set_title('Absolute Accuracy Improvement:\nglobal_25 - local_25',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Mutation Count Bin', fontsize=10)
    ax.set_ylabel('Tumor Type', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, 'fig8_absolute_improvement_heatmap_global_vs_local_25.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Figure 8: Absolute Improvement Heatmap (600 DPI PNG)")
else:
    print("Warning: Could not create Figure 8 - tumor type data not available")

