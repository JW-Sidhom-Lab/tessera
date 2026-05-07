"""Per-variant accuracy plots from GENIE inference results.

Reads the pickle files written by ``get_variant_loss_acc.py`` and
renders the per-cohort and per-variant-burden accuracy panels backing
Fig. 1 f-g.

Figures generated:

1. Per-variant accuracy by sequence type
2. Per-variant accuracy by variant burden
3. Per-variant accuracy by tumor type (if CANCER_TYPE column available)
"""

import pandas as pd
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducible bootstrap sampling
np.random.seed(42)

# ============================================================================
# Configuration
# ============================================================================

LOSS_FILES_DIR = 'var_loss/'  # Directory containing loss files (.pkl) - modify this to change the directory
# Publication-quality figures with 95% confidence intervals
USE_BOOTSTRAP = True
n_bootstrap = 1000 if USE_BOOTSTRAP else 0

plot_dir = 'plots/accuracy_analysis'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# ============================================================================
# Load GENIE Data
# ============================================================================

print("Loading model inference results on GENIE data...")
files = os.listdir(LOSS_FILES_DIR)
accuracy_data = []
genie_data = None  # Will be loaded from first pickle file

for file in files:
    if file.endswith('_genie_loss_variant.pkl'):
        # Extract model name from filename (e.g., TCGA_PanCan_SNV_global_25 -> global_25)
        model_name_full = file.replace('_genie_loss_variant.pkl', '')
        try:
            # Get last two parts (context type and context length)
            parts = model_name_full.split('_')
            model_name = '_'.join([parts[3], parts[4]])
        except (IndexError, ValueError):
            # Handle baseline case
            model_name = 'baseline_loss'

        print(f"  Processing: {model_name_full} -> {model_name}")

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            results = pickle.load(f)

        # Load GENIE data from first file (same for all models)
        if genie_data is None:
            genie_data = results['genie_data']
            print(f"  Loaded GENIE data: {len(genie_data)} variants from {genie_data['Tumor_Sample_Barcode'].nunique()} samples")

        # Extract predictions and ground truth
        y_pred = results['y_pred']
        y_true = results['y_true']
        y_pred_ref = results['y_pred_ref']
        y_true_ref = results['y_true_ref']

        # Compute accuracy metrics
        # ALT accuracy: all ALT sequence positions predicted correctly
        pred_classes_alt = np.argmax(y_pred, axis=-1)
        acc_alt = np.all(pred_classes_alt == y_true, axis=1).astype(int)

        # REF accuracy: all REF sequence positions predicted correctly
        pred_classes_ref = np.argmax(y_pred_ref, axis=-1)
        acc_ref = np.all(pred_classes_ref == y_true_ref, axis=1).astype(int)

        # Combined accuracy: BOTH ALT and REF must be correct
        acc_combined = (acc_alt & acc_ref).astype(int)

        # Store results by type
        for acc_val, acc_type in [(acc_combined, 'Combined'), (acc_ref, 'Ref'), (acc_alt, 'Alt')]:
            accuracy_data.append(pd.DataFrame({
                'Model': model_name,
                'Accuracy': acc_val,
                'Type': acc_type
            }))

if len(accuracy_data) == 0:
    print(f"ERROR: No model results found in {LOSS_FILES_DIR}")
    print("Please run get_variant_loss_acc.py first or check that output files exist.")
    exit(1)

# Convert to DataFrame
df = pd.concat(accuracy_data, ignore_index=True)

print("\nAccuracy data loaded successfully!")
print(f"Models found: {df['Model'].unique()}")
print(f"Shape: {df.shape}")

# ============================================================================
# MANUSCRIPT FIGURES - Accuracy Analysis
# ============================================================================

# Define model order matching original tcga_pancan_snv analysis
order = ['baseline_loss', 'local_1', 'global_1', 'local_10', 'global_10', 'local_25', 'global_25']

# Filter to only models that exist in the data
order = [m for m in order if m in df['Model'].unique()]

# Define colorblind-safe color palette (Paul Tol scheme - publication quality)
# Maintains semantic grouping: baseline gray, context families with light/dark pairs
# Passes deuteranopia, protanopia, and tritanopia tests
color_palette = {
    'baseline_loss': '#777777',  # Gray baseline
    'local_1': '#88CCEE',        # Cyan (light)
    'global_1': '#0077BB',       # Blue (dark)
    'local_10': '#DDAA33',       # Orange (light)
    'global_10': '#CC3311',      # Red (dark)
    'local_25': '#AA3377',       # Magenta (light)
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

# ============================================================================
# Figure 1: Per-Variant Accuracy by Sequence Type
# ============================================================================

# Calculate mean accuracy and optionally bootstrap for confidence intervals
bootstrap_results = []

for model in order:
    for acc_type in ['Combined', 'Ref', 'Alt']:
        # Get accuracy values for this model and type
        mask = (df['Model'] == model) & (df['Type'] == acc_type)
        acc_values = df[mask]['Accuracy'].values

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

print("✓ Figure 1: Per-Variant Accuracy by Sequence Type (600 DPI PNG)")

# Compute pairwise statistics comparing models within each sequence type
from scipy import stats
pairwise_stats = []

for acc_type in type_data:  # Compare models within each sequence type
    # Get all pairwise comparisons between models
    for i, model1 in enumerate(order):
        for model2 in order[i+1:]:
            mask1 = (df['Model'] == model1) & (df['Type'] == acc_type)
            mask2 = (df['Model'] == model2) & (df['Type'] == acc_type)

            acc_values1 = df[mask1]['Accuracy'].values
            acc_values2 = df[mask2]['Accuracy'].values

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
# Figure 2: Per-Variant Accuracy by Variant Burden
# ============================================================================

# Set up bin edges and labels for fixed binning (adjusted for GENIE data)
bin_edges = [1, 25, 50, 75, 100, np.inf]
bin_labels = ['1-25', '25-50', '50-75', '75-100', '100+']
ordered_bins = bin_labels

# Count variants per sample from the original data
variant_counts_per_sample = genie_data.groupby('Tumor_Sample_Barcode').size().reset_index(name='Variant_Count')

# Prepare data: add variant count bins to the per-variant accuracy data
per_variant_data_with_bins = []

for file in files:
    if file.endswith('_genie_loss_variant.pkl'):
        model_name_full = file.replace('_genie_loss_variant.pkl', '')
        try:
            parts = model_name_full.split('_')
            model_name = '_'.join([parts[3], parts[4]])
        except (IndexError, ValueError):
            model_name = 'baseline_loss'

        with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
            results = pickle.load(f)

        y_pred = results['y_pred']
        y_true = results['y_true']
        y_pred_ref = results['y_pred_ref']
        y_true_ref = results['y_true_ref']

        # Calculate combined accuracy (both alt and ref correct)
        pred_classes_alt = np.argmax(y_pred, axis=-1)
        acc_alt = np.all(pred_classes_alt == y_true, axis=1).astype(int)

        pred_classes_ref = np.argmax(y_pred_ref, axis=-1)
        acc_ref = np.all(pred_classes_ref == y_true_ref, axis=1).astype(int)

        acc_combined = (acc_alt & acc_ref).astype(int)

        # Vectorized: Create temporary dataframe and merge with variant counts
        temp_df = pd.DataFrame({
            'Tumor_Sample_Barcode': genie_data['Tumor_Sample_Barcode'].values,
            'Model': model_name,
            'Accuracy': acc_combined
        })

        # Merge with variant counts
        temp_df = temp_df.merge(variant_counts_per_sample, on='Tumor_Sample_Barcode', how='left')

        # Assign to bin using pandas cut (vectorized)
        temp_df['Bin'] = pd.cut(
            temp_df['Variant_Count'],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=False
        )

        # Drop rows with missing bins and append
        temp_df = temp_df.dropna(subset=['Bin'])
        per_variant_data_with_bins.append(temp_df[['Model', 'Accuracy', 'Bin']])

# Convert to DataFrame (concatenate all model dataframes)
df_per_variant_bins = pd.concat(per_variant_data_with_bins, ignore_index=True)

# Filter to bins with data
bins_with_data = [b for b in ordered_bins if b in df_per_variant_bins['Bin'].unique()]

print(f"\nPer-variant accuracy by variant burden - Bins to plot: {bins_with_data}")

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
        bars = ax.bar(positions, means, width, label=display_names[model],
                      color=color_palette.get(model, '#808080'), alpha=0.85,
                      yerr=[errors_lower, errors_upper], capsize=2, error_kw={'linewidth': 1, 'alpha': 0.7})
    else:
        bars = ax.bar(positions, means, width, label=display_names[model],
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
plt.savefig(os.path.join(plot_dir, 'fig2_per_variant_accuracy_by_variant_burden.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Figure 2: Per-Variant Accuracy by Variant Burden (600 DPI PNG)")

# ============================================================================
# Figure 3: Per-Variant Accuracy by Tumor Type (if available)
# ============================================================================

# Get cancer type information from CANCER_TYPE column
col_type = 'CANCER_TYPE'

if col_type in genie_data.columns:
    # Prepare data: add cancer type to per-variant accuracy data
    per_variant_data_with_cancer = []

    for file in files:
        if file.endswith('_genie_loss_variant.pkl'):
            model_name_full = file.replace('_genie_loss_variant.pkl', '')
            try:
                parts = model_name_full.split('_')
                model_name = '_'.join([parts[3], parts[4]])
            except (IndexError, ValueError):
                model_name = 'baseline_loss'

            with open(os.path.join(LOSS_FILES_DIR, file), 'rb') as f:
                results = pickle.load(f)

            y_pred = results['y_pred']
            y_true = results['y_true']
            y_pred_ref = results['y_pred_ref']
            y_true_ref = results['y_true_ref']

            # Calculate combined accuracy (both alt and ref correct)
            pred_classes_alt = np.argmax(y_pred, axis=-1)
            acc_alt = np.all(pred_classes_alt == y_true, axis=1).astype(int)

            pred_classes_ref = np.argmax(y_pred_ref, axis=-1)
            acc_ref = np.all(pred_classes_ref == y_true_ref, axis=1).astype(int)

            acc_combined = (acc_alt & acc_ref).astype(int)

            # Vectorized: Create temporary dataframe and merge with cancer types
            temp_df = pd.DataFrame({
                'Tumor_Sample_Barcode': genie_data['Tumor_Sample_Barcode'].values,
                'Model': model_name,
                'Accuracy': acc_combined
            })

            # Add cancer type information
            sample_cancer_types = genie_data.groupby('Tumor_Sample_Barcode')[col_type].first().reset_index()
            sample_cancer_types.columns = ['Tumor_Sample_Barcode', 'cancer_type']
            temp_df = temp_df.merge(sample_cancer_types, on='Tumor_Sample_Barcode', how='left')

            # Drop rows with missing cancer types and append
            temp_df = temp_df.dropna(subset=['cancer_type'])
            per_variant_data_with_cancer.append(temp_df[['Model', 'Accuracy', 'cancer_type']].rename(columns={'cancer_type': 'Cancer_Type'}))

    if per_variant_data_with_cancer:
        # Convert to DataFrame (concatenate all model dataframes)
        df_per_variant_cancer = pd.concat(per_variant_data_with_cancer, ignore_index=True)

        # Sort cancer types by per-variant mean accuracy (across all models and variants)
        cancer_type_per_variant_mean = df_per_variant_cancer.groupby('Cancer_Type')['Accuracy'].mean().sort_values(ascending=False)
        cancer_types_to_plot = cancer_type_per_variant_mean.index.tolist()

        print(f"\nPer-variant accuracy by cancer type - Types to plot: {len(cancer_types_to_plot)}")
        print(f"Top 5 cancer types by per-variant accuracy: {cancer_types_to_plot[:5]}")

        # Calculate mean accuracy and optionally bootstrap for confidence intervals
        bootstrap_results_cancer = []

        for model in order:
            for cancer_type in cancer_types_to_plot:
                # Get accuracy values for this model and cancer type
                mask = (df_per_variant_cancer['Model'] == model) & (df_per_variant_cancer['Cancer_Type'] == cancer_type)
                acc_values = df_per_variant_cancer[mask]['Accuracy'].values

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

                bootstrap_results_cancer.append({
                    'Model': model,
                    'Cancer_Type': cancer_type,
                    'Mean_Accuracy': mean_acc,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'Error_Lower': mean_acc - ci_lower,
                    'Error_Upper': ci_upper - mean_acc
                })

        df_bootstrap_cancer = pd.DataFrame(bootstrap_results_cancer)

        # Create publication-quality grouped bar plot
        fig, ax = plt.subplots(figsize=(14.0, 5.0))

        x = np.arange(len(cancer_types_to_plot))
        width = 0.11
        n_models = len(order)

        for i, model in enumerate(order):
            model_data = df_bootstrap_cancer[df_bootstrap_cancer['Model'] == model]

            positions = x + (i - n_models/2 + 0.5) * width
            means = [model_data[model_data['Cancer_Type'] == ct]['Mean_Accuracy'].values[0]
                     if len(model_data[model_data['Cancer_Type'] == ct]) > 0 else 0
                     for ct in cancer_types_to_plot]
            errors_lower = [model_data[model_data['Cancer_Type'] == ct]['Error_Lower'].values[0]
                            if len(model_data[model_data['Cancer_Type'] == ct]) > 0 else 0
                            for ct in cancer_types_to_plot]
            errors_upper = [model_data[model_data['Cancer_Type'] == ct]['Error_Upper'].values[0]
                            if len(model_data[model_data['Cancer_Type'] == ct]) > 0 else 0
                            for ct in cancer_types_to_plot]

            if USE_BOOTSTRAP:
                bars = ax.bar(positions, means, width, label=display_names[model],
                              color=color_palette.get(model, '#808080'), alpha=0.85,
                              yerr=[errors_lower, errors_upper], capsize=2, error_kw={'linewidth': 1, 'alpha': 0.7})
            else:
                bars = ax.bar(positions, means, width, label=display_names[model],
                              color=color_palette.get(model, '#808080'), alpha=0.85)

        ax.set_xlabel('Cancer Type', fontsize=11)
        ax.set_ylabel('Per-Variant Accuracy', fontsize=11)
        ax.set_title(f'Per-Variant Accuracy by Tumor Type',
                     fontsize=24, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(cancer_types_to_plot, rotation=45, ha='right', fontsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # Add horizontal grid
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        ax.set_axisbelow(True)

        # Legend
        ax.legend(
            title='Model',
            fontsize=9,
            title_fontsize=10,
            loc='upper right',
            ncol=2,
            frameon=True,
            framealpha=0.95,
            edgecolor='gray'
        )

        ax.set_ylim([0, 1])
        sns.despine()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'fig3_per_variant_accuracy_by_tumor_type.png'), dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

        print("✓ Figure 3: Per-Variant Accuracy by Tumor Type (600 DPI PNG)")
    else:
        print("Warning: No cancer type data found for Figure 3. Skipping.")
else:
    print(f"Warning: Could not find '{col_type}' column. Skipping Figure 3.")

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)
