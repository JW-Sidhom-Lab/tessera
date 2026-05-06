"""
SNV+CNA Combined Tumor Type Classifier - Ensemble with Macro-Fold Nested CV.

This script trains an ensemble of tumor type classifiers using k-fold cross-validation
with learned representations from BOTH SNV (single nucleotide variants) and CNA
(copy number alterations).

Features used:
1. SNV features - aggregated per sample using weighted average + max pooling
2. CNA features - aggregated per sample using mean + max pooling
3. Concatenation of both feature types for multimodal prediction

Training Strategy:
- Macro-Fold Nested CV (default):
  - Splits data into N_MACRO_FOLDS outer folds
  - Each outer fold is held out as a test set
  - Inner N_INNER_FOLDS CV is performed on remaining data
  - Each sample is tested exactly once across all macro-folds

Configuration:
- SNV_MODEL_NAME: Name of the SNV model to load features from
  Options: baseline, global_1, global_10, global_25, local_1, local_10, local_25
- CNA_MODEL_NAME: Name of the CNA model to load features from
  Options: attn_0, attn_1, attn_2
- N_MACRO_FOLDS: Number of outer folds for macro-fold nested CV (default: 5)
- N_INNER_FOLDS: Number of inner folds for CV (default: 10)

Usage:
    python tumor_type_classifier_snv_cna.py

Outputs:
    - Trained MLP models (.keras files)
    - Label encoder (.pkl)
    - Ensemble classification results (.pkl)
    - Visualizations (ROC curves, confusion matrices)
"""

import os
import gc
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
import matplotlib
_MPL_BACKEND = os.environ.get("MATPLOTLIB_BACKEND")
if _MPL_BACKEND:
    matplotlib.use(_MPL_BACKEND, force=True)
import matplotlib.pyplot as plt
import seaborn as sns


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


# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
# Manuscript multimodal classifier = global_25 SNV features + attn_2 CNA features.
SNV_MODEL_NAME = os.environ.get("SNV_MODEL_NAME", "global_25")
CNA_MODEL_NAME = os.environ.get("CNA_MODEL_NAME", "attn_2")

# Cross-validation + control flags.
SHUFFLE_LABELS = os.environ.get("SHUFFLE_LABELS", "0") != "0"  # control experiment
N_FOLDS = int(os.environ.get("N_FOLDS", "10"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))
USE_FULL_DATASET_CV = os.environ.get("USE_FULL_DATASET_CV", "0") != "0"
USE_MACRO_NESTED_CV = os.environ.get("USE_MACRO_NESTED_CV", "1") != "0"
test_size = float(os.environ.get("TEST_SIZE", "0.25"))  # only used if both above are False
N_MACRO_FOLDS = int(os.environ.get("N_MACRO_FOLDS", "5"))
N_INNER_FOLDS = int(os.environ.get("N_INNER_FOLDS", "10"))

# Class weighting + PCA + tumor-type filter.
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "0") != "0"
APPLY_PCA = os.environ.get("APPLY_PCA", "0") != "0"
PCA_N_COMPONENTS = (int(os.environ["PCA_N_COMPONENTS"])
                    if os.environ.get("PCA_N_COMPONENTS") else None)
PCA_EXPLAINED_VARIANCE_THRESHOLD = float(
    os.environ.get("PCA_EXPLAINED_VARIANCE_THRESHOLD", "0.95"))
MIN_SAMPLES_PER_TYPE = int(os.environ.get("MIN_SAMPLES_PER_TYPE", "100"))
TOP_N_TUMOR_TYPES = (int(os.environ["TOP_N_TUMOR_TYPES"])
                     if os.environ.get("TOP_N_TUMOR_TYPES") else None)

# Paths (relative to scripts/tumor_classification_snv_cna/).
SNV_FEATURES_DIR = os.environ.get("SNV_FEATURES_DIR", "../tcga_pancan_snv/var_features")
CNA_FEATURES_DIR = os.environ.get("CNA_FEATURES_DIR", "../tcga_pancan_cna/cna_features")
DATA_DIR = os.environ.get("DATA_DIR", "../data/tcga")
CLINICAL_PATH = os.environ.get("CLINICAL_PATH", "../../data/TCGA_PanCan/clinical.csv")
MODELS_DIR = os.environ.get("MODELS_DIR", "models_macro")
PLOTS_DIR = os.environ.get("PLOTS_DIR", "plots")

# Create plots and models directories
plots_dir = PLOTS_DIR
os.makedirs(plots_dir, exist_ok=True)

models_dir = MODELS_DIR
os.makedirs(models_dir, exist_ok=True)

# Combined model identifier
COMBINED_MODEL_NAME = f'{SNV_MODEL_NAME}_{CNA_MODEL_NAME}'

print("=" * 80)
print("SNV+CNA COMBINED TUMOR TYPE CLASSIFIER - ENSEMBLE")
print("=" * 80)
print(f"SNV model: {SNV_MODEL_NAME}")
print(f"CNA model: {CNA_MODEL_NAME}")
print(f"Combined model name: {COMBINED_MODEL_NAME}")

if USE_MACRO_NESTED_CV:
    print(f"Ensemble strategy: {N_MACRO_FOLDS}-fold macro nested CV ({N_INNER_FOLDS} inner folds)")
    print(f"Evaluation mode: Macro-fold nested cross-validation")
    print(f"  - Outer folds: {N_MACRO_FOLDS} (each held out as test set)")
    print(f"  - Inner folds: {N_INNER_FOLDS} (on remaining {(N_MACRO_FOLDS-1)/N_MACRO_FOLDS*100:.0f}% of data)")
elif USE_FULL_DATASET_CV:
    print(f"Ensemble strategy: {N_FOLDS}-fold cross-validation")
    print(f"Evaluation mode: Full dataset CV (out-of-fold predictions)")
else:
    print(f"Ensemble strategy: {N_FOLDS}-fold cross-validation")
    print(f"Evaluation mode: Separate test set ({test_size*100:.0f}% held out)")
print(f"Models directory: {models_dir}")
if APPLY_PCA:
    print(f"PCA: Enabled (variance threshold: {PCA_EXPLAINED_VARIANCE_THRESHOLD})")
print("")


# ============================================================================
# 1. Load SNV Features
# ============================================================================
print("\n" + "=" * 60)
print("1. LOADING SNV FEATURES")
print("=" * 60)

snv_features_path = os.path.join(SNV_FEATURES_DIR, f'TCGA_PanCan_SNV_{SNV_MODEL_NAME}_features.pkl')
print(f"Loading SNV features from: {snv_features_path}")

with open(snv_features_path, 'rb') as f:
    snv_features_out = pickle.load(f)

snv_features_train = snv_features_out['train']
snv_features_valid = snv_features_out['valid']

# Load SNV variant data with sample IDs and tumor types
snv_train_data = pd.read_csv(os.path.join(DATA_DIR, 'train_data_snv.csv'))
snv_valid_data = pd.read_csv(os.path.join(DATA_DIR, 'valid_data_snv.csv'))

# Combine train and valid
snv_features = np.vstack([snv_features_train, snv_features_valid])
snv_data = pd.concat([snv_train_data, snv_valid_data], axis=0).reset_index(drop=True)

print(f"Loaded {snv_features.shape[0]:,} SNV variant features")
print(f"  Train: {snv_features_train.shape[0]:,}")
print(f"  Valid: {snv_features_valid.shape[0]:,}")
print(f"  Feature dimensionality: {snv_features.shape[1]}")

# Extract short sample ID (first 15 chars) for aggregation
snv_data['Sample_ID_Short'] = snv_data['Tumor_Sample_Barcode'].str[:15]

# Combine COAD+READ → COADREAD and ESCA+STAD → ESCASTAD
snv_data['type'] = snv_data['type'].replace({
    'COAD': 'COADREAD', 'READ': 'COADREAD',
    'ESCA': 'ESCASTAD', 'STAD': 'ESCASTAD',
})

# Clean up train/valid splits immediately after combining
print("  Cleaning up train/valid splits...")
del snv_features_train, snv_features_valid, snv_features_out
gc.collect()


# ============================================================================
# 2. Load CNA Features
# ============================================================================
print("\n" + "=" * 60)
print("2. LOADING CNA FEATURES")
print("=" * 60)

cna_features_path = os.path.join(CNA_FEATURES_DIR, f'TCGA_PanCan_CNA_{CNA_MODEL_NAME}_cna_features.pkl')
print(f"Loading CNA features from: {cna_features_path}")

with open(cna_features_path, 'rb') as f:
    cna_features_out = pickle.load(f)

# Extract CNA features and metadata
cna_features_train = cna_features_out['train_features_cna']
cna_features_valid = cna_features_out['valid_features_cna']
cna_train_data = cna_features_out['train_data_cna']
cna_valid_data = cna_features_out['valid_data_cna']

# Combine train and valid
cna_features = np.vstack([cna_features_train, cna_features_valid])
cna_data = pd.concat([cna_train_data, cna_valid_data], axis=0).reset_index(drop=True)

print(f"Loaded {cna_features.shape[0]:,} CNA segment features")
print(f"  Train: {cna_features_train.shape[0]:,}")
print(f"  Valid: {cna_features_valid.shape[0]:,}")
print(f"  Feature dimensionality: {cna_features.shape[1]}")

# Load tumor type labels from clinical data
print("\n  Loading tumor types from clinical data...")
clinical = pd.read_csv(CLINICAL_PATH)
print(f"  Loaded clinical data: {len(clinical):,} patients, {clinical['type'].nunique()} tumor types")

# Extract patient ID from sample barcode (first 12 characters)
cna_data['patient_barcode'] = cna_data['Tumor_Sample_Barcode'].str[:12]

# Create tumor type mapping from clinical data
tumor_type_mapping = clinical[['bcr_patient_barcode', 'type']].copy()
tumor_type_mapping.columns = ['patient_barcode', 'type']

# Merge tumor types with CNA data
cna_data = cna_data.merge(
    tumor_type_mapping,
    on='patient_barcode',
    how='left'
)

# Combine COAD+READ → COADREAD and ESCA+STAD → ESCASTAD
cna_data['type'] = cna_data['type'].replace({
    'COAD': 'COADREAD', 'READ': 'COADREAD',
    'ESCA': 'ESCASTAD', 'STAD': 'ESCASTAD',
})

# Check for missing tumor types in CNA data
missing_types = cna_data['type'].isna().sum()
if missing_types > 0:
    print(f"  WARNING: {missing_types} CNA segments without tumor type labels")
    print(f"  Removing segments without tumor type...")
    has_type_mask = ~cna_data['type'].isna()
    cna_data = cna_data[has_type_mask].reset_index(drop=True)
    cna_features = cna_features[has_type_mask.values]
    print(f"  Remaining segments: {len(cna_data):,}")

# Extract short sample ID (first 15 chars) for aggregation
cna_data['Sample_ID_Short'] = cna_data['Tumor_Sample_Barcode'].str[:15]

# Clean up train/valid splits
print("  Cleaning up train/valid splits...")
del cna_features_train, cna_features_valid, cna_features_out, clinical, tumor_type_mapping
gc.collect()


# ============================================================================
# 3. Find Common Samples Between SNV and CNA
# ============================================================================
print("\n" + "=" * 60)
print("3. FINDING COMMON SAMPLES")
print("=" * 60)

# Get unique samples from each modality
snv_samples = set(snv_data['Sample_ID_Short'].unique())
cna_samples = set(cna_data['Sample_ID_Short'].unique())

print(f"Unique samples with SNV data: {len(snv_samples):,}")
print(f"Unique samples with CNA data: {len(cna_samples):,}")

# Find intersection
common_samples = snv_samples & cna_samples
print(f"Common samples (intersection): {len(common_samples):,}")

# Filter both datasets to common samples
snv_data_filtered = snv_data[snv_data['Sample_ID_Short'].isin(common_samples)].reset_index(drop=True)
snv_features_filtered = snv_features[snv_data['Sample_ID_Short'].isin(common_samples).values]

cna_data_filtered = cna_data[cna_data['Sample_ID_Short'].isin(common_samples)].reset_index(drop=True)
cna_features_filtered = cna_features[cna_data['Sample_ID_Short'].isin(common_samples).values]

print(f"\nAfter filtering to common samples:")
print(f"  SNV variants: {len(snv_data_filtered):,}")
print(f"  CNA segments: {len(cna_data_filtered):,}")

# Clean up
del snv_features, cna_features, snv_data, cna_data
gc.collect()


# ============================================================================
# 4. Filter for Valid Tumor Types
# ============================================================================
print("\n" + "=" * 60)
print("4. FILTERING TUMOR TYPES")
print("=" * 60)

# Get tumor type counts from SNV data (has tumor types)
data_tumor_type = snv_data_filtered[['Sample_ID_Short', 'type']].drop_duplicates()
data_tumor_type_counts = data_tumor_type['type'].value_counts()

print(f"Total unique tumor types in common samples: {len(data_tumor_type_counts)}")
print(f"Total common samples: {len(data_tumor_type):,}")

# Filter tumor types by count
if TOP_N_TUMOR_TYPES is not None:
    valid_tumor_types = data_tumor_type_counts.nlargest(TOP_N_TUMOR_TYPES).index.tolist()
    print(f"Selected top {TOP_N_TUMOR_TYPES} most frequent tumor types")
elif MIN_SAMPLES_PER_TYPE is not None:
    valid_tumor_types = data_tumor_type_counts[data_tumor_type_counts >= MIN_SAMPLES_PER_TYPE].index.tolist()
    print(f"Selected tumor types with >= {MIN_SAMPLES_PER_TYPE} samples")
else:
    valid_tumor_types = data_tumor_type_counts.index.tolist()
    print("Using all tumor types")

print(f"Valid tumor types: {len(valid_tumor_types)}")
for tt in sorted(valid_tumor_types):
    print(f"  {tt}: {data_tumor_type_counts[tt]:,} samples")

# Get valid samples
valid_samples = data_tumor_type[data_tumor_type['type'].isin(valid_tumor_types)]['Sample_ID_Short'].unique()
print(f"\nValid samples after tumor type filtering: {len(valid_samples):,}")

# Filter SNV data
idx_keep_snv = snv_data_filtered['Sample_ID_Short'].isin(valid_samples)
snv_data_filtered = snv_data_filtered[idx_keep_snv].reset_index(drop=True)
snv_features_filtered = snv_features_filtered[idx_keep_snv.values]

# Filter CNA data
idx_keep_cna = cna_data_filtered['Sample_ID_Short'].isin(valid_samples)
cna_data_filtered = cna_data_filtered[idx_keep_cna].reset_index(drop=True)
cna_features_filtered = cna_features_filtered[idx_keep_cna.values]

print(f"SNV variants after filtering: {snv_features_filtered.shape[0]:,}")
print(f"CNA segments after filtering: {cna_features_filtered.shape[0]:,}")


# ============================================================================
# 5. Aggregate SNV Features per Sample
# ============================================================================
print("\n" + "=" * 60)
print("5. AGGREGATING SNV FEATURES PER SAMPLE")
print("=" * 60)

print("\nUsing weighted average + max pooling aggregation...")

# Scale SNV features before aggregation
scaler_snv = RobustScaler()
snv_features_scaled = scaler_snv.fit_transform(snv_features_filtered)

# Create DataFrame for aggregation using truncated sample IDs
snv_df = pd.DataFrame(snv_features_scaled)
snv_df.index = snv_data_filtered['Sample_ID_Short']
snv_df['vaf'] = 1.0  # Equal weighting (or use snv_data_filtered['vaf'] for VAF weighting)


def snv_weighted_average(group):
    """Compute weighted average of features per sample"""
    group_features = group.drop(columns=['vaf']).values
    vaf_weights = group['vaf'].values

    if len(vaf_weights) == group_features.shape[0]:
        weighted_avg = np.average(group_features, axis=0, weights=vaf_weights)
        return pd.Series(weighted_avg)
    else:
        raise ValueError("Mismatch between group size and VAF weights.")


def snv_max_pooling(group):
    """Compute max pooling of features per sample"""
    group_features = group.drop(columns=['vaf']).values
    max_features = np.max(group_features, axis=0)
    return pd.Series(max_features)


# Compute weighted average features
print("  Computing weighted average features...")
snv_agg_mean = snv_df.groupby(snv_df.index).apply(snv_weighted_average)

# Compute max pooled features
print("  Computing max pooled features...")
snv_agg_max = snv_df.groupby(snv_df.index).apply(snv_max_pooling)

# Concatenate mean and max features
snv_agg = pd.concat([snv_agg_mean, snv_agg_max], axis=1)
snv_agg.columns = [f'snv_mean_{i}' for i in range(snv_agg_mean.shape[1])] + \
                  [f'snv_max_{i}' for i in range(snv_agg_max.shape[1])]

# Add Total Mutational Burden as a feature (count of variants per sample)
snv_mutational_burden = snv_df.groupby(snv_df.index).size()
snv_agg['snv_mutational_burden'] = np.log1p(snv_mutational_burden)

print(f"Aggregated SNV features: {snv_agg.shape[0]:,} samples")
print(f"  Feature dimensionality: {snv_agg.shape[1]} (mean + max pooling + mutational burden)")
print(f"    - Mean-pooled features: {snv_agg_mean.shape[1]}")
print(f"    - Max-pooled features: {snv_agg_max.shape[1]}")
print(f"  Mean mutational burden: {snv_mutational_burden.mean():.1f} variants/sample")
print(f"  Median mutational burden: {snv_mutational_burden.median():.1f} variants/sample")
print(f"  Range: {snv_mutational_burden.min()}-{snv_mutational_burden.max()} variants/sample")

n_snv_features = snv_agg.shape[1]

# Clean up intermediate SNV aggregation variables
print("  Cleaning up SNV aggregation intermediates...")
del snv_features_scaled, snv_df, snv_agg_mean, snv_agg_max, snv_mutational_burden
gc.collect()


# ============================================================================
# 6. Aggregate CNA Features per Sample
# ============================================================================
print("\n" + "=" * 60)
print("6. AGGREGATING CNA FEATURES PER SAMPLE")
print("=" * 60)

print("\nUsing mean + max pooling aggregation (no weighting)...")

# Scale CNA features before aggregation
scaler_cna = RobustScaler()
cna_features_scaled = scaler_cna.fit_transform(cna_features_filtered)

# Create DataFrame for aggregation using truncated sample IDs
cna_df = pd.DataFrame(cna_features_scaled)
cna_df.index = cna_data_filtered['Sample_ID_Short']


def cna_simple_mean(group):
    """Compute simple mean of features per sample"""
    return pd.Series(np.mean(group.values, axis=0))


def cna_max_pooling(group):
    """Compute max pooling of features per sample"""
    return pd.Series(np.max(group.values, axis=0))


# Compute mean features
print("  Computing mean features...")
cna_agg_mean = cna_df.groupby(cna_df.index).apply(cna_simple_mean)

# Compute max pooled features
print("  Computing max pooled features...")
cna_agg_max = cna_df.groupby(cna_df.index).apply(cna_max_pooling)

# Concatenate mean and max features
cna_agg = pd.concat([cna_agg_mean, cna_agg_max], axis=1)
cna_agg.columns = [f'cna_mean_{i}' for i in range(cna_agg_mean.shape[1])] + \
                  [f'cna_max_{i}' for i in range(cna_agg_max.shape[1])]

# Add segment count as a feature (number of CNA segments per sample)
cna_segment_count = cna_df.groupby(cna_df.index).size()
cna_agg['cna_segment_count'] = np.log1p(cna_segment_count)

print(f"Aggregated CNA features: {cna_agg.shape[0]:,} samples")
print(f"  Feature dimensionality: {cna_agg.shape[1]} (mean + max pooling + segment count)")
print(f"    - Mean-pooled features: {cna_agg_mean.shape[1]}")
print(f"    - Max-pooled features: {cna_agg_max.shape[1]}")
print(f"  Mean segment count: {cna_segment_count.mean():.1f} segments/sample")
print(f"  Median segment count: {cna_segment_count.median():.1f} segments/sample")
print(f"  Range: {cna_segment_count.min()}-{cna_segment_count.max()} segments/sample")

n_cna_features = cna_agg.shape[1]

# Clean up intermediate CNA aggregation variables
print("  Cleaning up CNA aggregation intermediates...")
del cna_features_scaled, cna_df, cna_agg_mean, cna_agg_max, cna_segment_count
gc.collect()


# ============================================================================
# 7. Concatenate SNV and CNA Features
# ============================================================================
print("\n" + "=" * 60)
print("7. CONCATENATING SNV AND CNA FEATURES")
print("=" * 60)

# Ensure both have the same samples (should already be the case)
common_samples_agg = set(snv_agg.index) & set(cna_agg.index)
print(f"Samples with both SNV and CNA aggregated features: {len(common_samples_agg):,}")

# Filter to common samples and align indices
snv_agg = snv_agg.loc[snv_agg.index.isin(common_samples_agg)]
cna_agg = cna_agg.loc[cna_agg.index.isin(common_samples_agg)]

# Sort by index to ensure alignment
snv_agg = snv_agg.sort_index()
cna_agg = cna_agg.sort_index()

# Verify alignment
assert list(snv_agg.index) == list(cna_agg.index), "Sample indices do not match!"

# Concatenate features
combined_agg = pd.concat([snv_agg, cna_agg], axis=1)

print(f"\nCombined feature matrix:")
print(f"  Samples: {combined_agg.shape[0]:,}")
print(f"  SNV features: {n_snv_features}")
print(f"  CNA features: {n_cna_features}")
print(f"  Total features: {combined_agg.shape[1]}")


# ============================================================================
# 8. Prepare Final Feature Matrix
# ============================================================================
print("\n" + "=" * 60)
print("8. PREPARING FINAL FEATURE MATRIX")
print("=" * 60)

# Get sample IDs
sample_ids_list = list(combined_agg.index)
n_samples = len(sample_ids_list)

print(f"Total samples with combined features: {n_samples:,}")
print(f"Feature dimensionality: {combined_agg.shape[1]}")


# ============================================================================
# 8b. Optional PCA on Aggregated Features
# ============================================================================
if APPLY_PCA:
    print("\n" + "=" * 60)
    print("8b. APPLYING PCA TO AGGREGATED FEATURES")
    print("=" * 60)

    print(f"\nOriginal dimensionality: {combined_agg.shape[1]}")

    # Standardize features before PCA
    scaler_pca = StandardScaler()
    features_for_pca = scaler_pca.fit_transform(combined_agg.values)

    if PCA_N_COMPONENTS is None:
        # Use explained variance threshold to determine number of components
        print(f"Using explained variance threshold: {PCA_EXPLAINED_VARIANCE_THRESHOLD}")

        # Fit PCA with all components first to determine optimal n_components
        pca_full = PCA(random_state=RANDOM_STATE)
        pca_full.fit(features_for_pca)

        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= PCA_EXPLAINED_VARIANCE_THRESHOLD) + 1

        print(f"Selected {n_components} components (explains {cumsum_var[n_components-1]:.2%} of variance)")

        # Apply PCA with selected number of components
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        final_features = pca.fit_transform(features_for_pca)
    else:
        n_components = PCA_N_COMPONENTS
        print(f"Using fixed {n_components} PCA components")

        # Fit and transform in one step
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        final_features = pca.fit_transform(features_for_pca)

        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"Explained variance: {explained_var:.2%}")

    print(f"PCA applied: {final_features.shape[1]} dimensions")
    pca_n_components_saved = final_features.shape[1]
else:
    final_features = combined_agg.values
    pca_n_components_saved = None
    print("Skipping PCA - using original aggregated features")


# ============================================================================
# 9. Prepare Labels
# ============================================================================
print("\n" + "=" * 60)
print("9. PREPARING LABELS")
print("=" * 60)

# Get tumor types for samples using truncated IDs (from SNV data)
tumor_type_map = snv_data_filtered[['Sample_ID_Short', 'type']].drop_duplicates()
tumor_type_map = tumor_type_map.set_index('Sample_ID_Short')['type']

# Get labels for samples
y = tumor_type_map.loc[sample_ids_list].values

# Remove any samples with missing labels
valid_label_mask = ~pd.isna(y)
final_features = final_features[valid_label_mask]
y = y[valid_label_mask]
sample_ids_final = np.array(sample_ids_list)[valid_label_mask]

print(f"Final dataset: {len(y):,} samples")
print(f"Number of unique tumor types: {len(np.unique(y))}")

# Optional: Shuffle labels for control experiment
if SHUFFLE_LABELS:
    print("\n" + "=" * 60)
    print("WARNING: LABELS ARE SHUFFLED - THIS IS A CONTROL EXPERIMENT")
    print("=" * 60)
    np.random.seed(RANDOM_STATE)
    y_shuffled = y.copy()
    np.random.shuffle(y_shuffled)
    y = y_shuffled
    print("Labels have been randomly shuffled to test for overfitting")

# Encode tumor types
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
n_classes = len(label_encoder.classes_)

# Final standardization of features
scaler_final = RobustScaler()
final_features = scaler_final.fit_transform(final_features)

print(f"\nTumor type distribution:")
tumor_type_counts = pd.Series(y).value_counts().sort_index()
for tumor_type, count in tumor_type_counts.head(10).items():
    print(f"  {tumor_type}: {count}")
if len(tumor_type_counts) > 10:
    print(f"  ... and {len(tumor_type_counts) - 10} more")


# ============================================================================
# 10. Train/Test Split (or use full dataset for CV)
# ============================================================================
print("\n" + "=" * 60)
print("10. SPLITTING DATA")
print("=" * 60)

if USE_MACRO_NESTED_CV:
    # Use macro-fold nested CV
    X_train = final_features
    y_train = y_encoded
    X_test = None
    y_test = None

    print(f"Using macro-fold nested cross-validation")
    print(f"Total samples: {X_train.shape[0]:,}")
    print(f"Feature dimensionality: {X_train.shape[1]}")
    print(f"Evaluation will be across all samples (each tested exactly once)")

elif USE_FULL_DATASET_CV:
    # Use entire dataset for cross-validation
    X_train = final_features
    y_train = y_encoded
    X_test = None
    y_test = None

    print(f"Using full dataset for cross-validation")
    print(f"Total samples: {X_train.shape[0]:,}")
    print(f"Feature dimensionality: {X_train.shape[1]}")
    print(f"Evaluation will be on out-of-fold predictions")
else:
    # Hold out a test set
    sample_indices = np.arange(len(final_features))

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        final_features, y_encoded, sample_indices,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Feature dimensionality: {X_train.shape[1]}")


# ============================================================================
# 11. Build and Train Ensemble
# ============================================================================
print("\n" + "=" * 60)
print("11. TRAINING ENSEMBLE")
print("=" * 60)


def create_mlp_model(input_dim, n_classes, dropout_rate=0.5):
    """Create MLP classifier for combined SNV+CNA features"""
    inputs = tf.keras.layers.Input(shape=(input_dim,))

    # First hidden layer
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Second hidden layer
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Third hidden layer
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Initialize cross-validation strategy
if USE_MACRO_NESTED_CV:
    n_folds_outer = N_MACRO_FOLDS
    n_folds_inner = N_INNER_FOLDS
    skf_outer = StratifiedKFold(n_splits=N_MACRO_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\nMACRO-FOLD NESTED CV: {N_MACRO_FOLDS} outer folds x {N_INNER_FOLDS} inner folds")
else:
    n_folds_outer = 1
    n_folds_inner = N_FOLDS

skf = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=RANDOM_STATE)

# Store models and training histories
ensemble_models = []
fold_histories = []
fold_val_accuracies = []
fold_val_losses = []

# For macro-fold mode: store test set predictions and indices
if USE_MACRO_NESTED_CV:
    macro_fold_predictions_list = []
    macro_fold_test_indices_list = []
    macro_fold_accuracies = []
else:
    # For full dataset CV mode: store out-of-fold predictions
    if USE_FULL_DATASET_CV:
        oof_predictions = np.zeros((len(X_train), n_classes))
        oof_indices = np.zeros(len(X_train), dtype=bool)

if USE_MACRO_NESTED_CV:
    print(f"Total samples: {X_train.shape[0]:,}")
    print(f"Samples per macro-fold (test): {X_train.shape[0] // N_MACRO_FOLDS:,}")
    print(f"Samples per macro-fold (train): {X_train.shape[0] * (N_MACRO_FOLDS - 1) // N_MACRO_FOLDS:,}")
else:
    print(f"\nTraining {N_FOLDS} models using stratified {N_FOLDS}-fold cross-validation")
    print(f"Training set size: {X_train.shape[0]:,} samples")
print("")

# Store feature stats before cleanup
n_combined_features = combined_agg.shape[1]
n_samples_total = len(y)
sample_ids_list_final = list(sample_ids_final)

# Clean up memory before training
print("Cleaning up memory before training...")
del snv_data_filtered, cna_data_filtered, snv_agg, cna_agg, combined_agg
del tumor_type_map, y
gc.collect()
print("Memory cleanup complete!")


# ============================================================================
# Main training loop
# ============================================================================

if USE_MACRO_NESTED_CV:
    # MACRO-FOLD NESTED CV: Outer loop over macro-folds
    print("\n" + "=" * 80)
    print("STARTING MACRO-FOLD NESTED CROSS-VALIDATION")
    print("=" * 80)

    for macro_idx, (macro_train_idx, macro_test_idx) in enumerate(skf_outer.split(X_train, y_train), 1):
        print("\n" + "=" * 80)
        print(f"MACRO-FOLD {macro_idx}/{N_MACRO_FOLDS}")
        print("=" * 80)

        # Split data for this macro-fold
        X_macro_train = X_train[macro_train_idx]
        y_macro_train = y_train[macro_train_idx]
        X_macro_test = X_train[macro_test_idx]
        y_macro_test = y_train[macro_test_idx]

        print(f"Macro-Fold {macro_idx}:")
        print(f"  Train: {X_macro_train.shape[0]:,} samples")
        print(f"  Test:  {X_macro_test.shape[0]:,} samples")

        # Reset ensemble for this macro-fold
        macro_ensemble_models = []
        macro_fold_val_accuracies = []

        # Inner K-fold CV on macro_train data
        skf_inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE + macro_idx)

        print(f"\n  Running {N_INNER_FOLDS}-fold inner CV...")

        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(skf_inner.split(X_macro_train, y_macro_train), 1):
            print(f"    Inner fold {inner_idx}/{N_INNER_FOLDS}...", end=" ")

            # Split data for this inner fold
            X_inner_train, X_inner_val = X_macro_train[inner_train_idx], X_macro_train[inner_val_idx]
            y_inner_train, y_inner_val = y_macro_train[inner_train_idx], y_macro_train[inner_val_idx]

            # Compute class weights if enabled
            class_weights = None
            if USE_CLASS_WEIGHTS:
                class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(y_inner_train),
                    y=y_inner_train
                )
                class_weights = {i: weight for i, weight in enumerate(class_weights)}

            # Create and compile model
            model = create_mlp_model(input_dim=X_macro_train.shape[1], n_classes=n_classes)
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Set up early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=0
            )

            # Train model
            history = model.fit(
                X_inner_train, y_inner_train,
                validation_data=(X_inner_val, y_inner_val),
                epochs=1000,
                batch_size=32,
                callbacks=[early_stopping],
                class_weight=class_weights,
                verbose=0
            )

            # Evaluate on inner fold validation set
            val_loss, val_accuracy = model.evaluate(X_inner_val, y_inner_val, verbose=0)
            macro_fold_val_accuracies.append(val_accuracy)

            print(f"Val Acc: {val_accuracy:.4f}")

            # Save model
            suffix = '_shuffled' if SHUFFLE_LABELS else ''
            model_path = os.path.join(models_dir, f'macro{macro_idx}_inner{inner_idx}_snv_cna{suffix}.keras')
            model.save(model_path)

            # Store model
            macro_ensemble_models.append(model)
            fold_histories.append(history.history)
            ensemble_models.append(model)

        # After inner CV, average ensemble predictions on macro_test
        print(f"\n  Generating ensemble predictions on macro-fold test set...")
        ensemble_probs = None
        for model in macro_ensemble_models:
            probs = model.predict(X_macro_test, verbose=0)
            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs

        ensemble_probs /= len(macro_ensemble_models)
        y_pred_macro = np.argmax(ensemble_probs, axis=1)
        macro_accuracy = accuracy_score(y_macro_test, y_pred_macro)

        print(f"  Mean Inner CV Accuracy: {np.mean(macro_fold_val_accuracies):.4f} +/- {np.std(macro_fold_val_accuracies):.4f}")
        print(f"  Ensemble Test Accuracy (Macro-Fold {macro_idx}): {macro_accuracy:.4f}")

        # Store results
        fold_val_accuracies.append(np.mean(macro_fold_val_accuracies))
        fold_val_losses.append(0)
        macro_fold_predictions_list.append(ensemble_probs)
        macro_fold_test_indices_list.append(macro_test_idx)
        macro_fold_accuracies.append(macro_accuracy)

    print("\n" + "=" * 80)
    print("MACRO-FOLD NESTED CV COMPLETE")
    print("=" * 80)
    print(f"\nMacro-Fold Accuracies: {[f'{acc:.4f}' for acc in macro_fold_accuracies]}")
    print(f"Mean: {np.mean(macro_fold_accuracies):.4f} +/- {np.std(macro_fold_accuracies):.4f}")

    # Reconstruct full dataset predictions
    print("\nReconstructing full dataset predictions from macro-folds...")

    full_predictions = np.zeros((len(X_train), n_classes))
    full_labels = np.zeros(len(X_train), dtype=int)

    for macro_idx, (pred, indices) in enumerate(zip(macro_fold_predictions_list, macro_fold_test_indices_list)):
        full_predictions[indices] = pred
        full_labels[indices] = y_train[indices]

    test_predictions = full_predictions
    y_test = full_labels
    test_pred_classes = np.argmax(full_predictions, axis=1)
    test_accuracy = accuracy_score(full_labels, test_pred_classes)

    print(f"Full dataset prediction accuracy: {test_accuracy:.4f}")
    print(f"Expected from macro-fold mean: {np.mean(macro_fold_accuracies):.4f}")

else:
    # Standard CV (non-macro-fold)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print("=" * 60)
        print(f"FOLD {fold_idx}/{N_FOLDS}")
        print("=" * 60)

        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        print(f"Fold {fold_idx} - Train: {X_fold_train.shape[0]:,} samples, Val: {X_fold_val.shape[0]:,} samples")

        class_weights = None
        if USE_CLASS_WEIGHTS:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_fold_train),
                y=y_fold_train
            )
            class_weights = {i: weight for i, weight in enumerate(class_weights)}

        model = create_mlp_model(input_dim=X_train.shape[1], n_classes=n_classes)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        if fold_idx == 1:
            print("\nModel architecture:")
            model.summary()
            print("")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=0
        )

        print(f"Training fold {fold_idx}...")
        history = model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=1000,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        val_loss, val_accuracy = model.evaluate(X_fold_val, y_fold_val, verbose=0)
        fold_val_accuracies.append(val_accuracy)
        fold_val_losses.append(val_loss)

        print(f"\nFold {fold_idx} Results:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")

        suffix = '_shuffled' if SHUFFLE_LABELS else ''
        model_path = os.path.join(models_dir, f'snv_cna_fold{fold_idx}{suffix}.keras')
        model.save(model_path)
        print(f"  Model saved: {model_path}")

        ensemble_models.append(model)
        fold_histories.append(history.history)

        if USE_FULL_DATASET_CV:
            val_predictions = model.predict(X_fold_val, verbose=0)
            oof_predictions[val_idx] = val_predictions
            oof_indices[val_idx] = True

        print("")


# Report ensemble training summary
print("=" * 60)
print("ENSEMBLE TRAINING COMPLETE")
print("=" * 60)
print(f"\nCross-Validation Results:")
print(f"  Mean Validation Accuracy: {np.mean(fold_val_accuracies):.4f} +/- {np.std(fold_val_accuracies):.4f}")
print(f"  Mean Validation Loss: {np.mean(fold_val_losses):.4f} +/- {np.std(fold_val_losses):.4f}")
print(f"\nPer-Fold Validation Accuracies:")
for i, acc in enumerate(fold_val_accuracies, 1):
    print(f"  Fold {i}: {acc:.4f}")
print("")


# ============================================================================
# 12. Evaluate Ensemble
# ============================================================================
print("\n" + "=" * 60)
if USE_MACRO_NESTED_CV:
    print("12. EVALUATING MACRO-FOLD NESTED CV RESULTS")
elif USE_FULL_DATASET_CV:
    print("12. EVALUATING ON OUT-OF-FOLD PREDICTIONS")
else:
    print("12. EVALUATING ENSEMBLE ON TEST SET")
print("=" * 60)

if USE_MACRO_NESTED_CV:
    print(f"Macro-fold nested CV results:")
    print(f"  Full dataset accuracy: {test_accuracy:.4f}")
    print(f"  Individual macro-fold accuracies: {[f'{acc:.4f}' for acc in macro_fold_accuracies]}")
    fold_predictions = None
    fold_test_accuracies = macro_fold_accuracies

elif USE_FULL_DATASET_CV:
    print(f"Using out-of-fold predictions for all {len(X_train):,} samples")

    if not np.all(oof_indices):
        missing = np.sum(~oof_indices)
        print(f"WARNING: {missing} samples do not have out-of-fold predictions!")

    test_predictions = oof_predictions
    test_pred_classes = np.argmax(oof_predictions, axis=1)
    y_test = y_train
    test_accuracy = accuracy_score(y_train, test_pred_classes)

    print(f"\nOut-of-fold accuracy: {test_accuracy:.4f}")
    fold_predictions = None
    fold_test_accuracies = fold_val_accuracies

else:
    print(f"Generating predictions from {len(ensemble_models)} models...")
    fold_predictions = []
    fold_test_accuracies = []

    for fold_idx, model in enumerate(ensemble_models, 1):
        fold_pred_probs = model.predict(X_test, verbose=0)
        fold_pred_classes = np.argmax(fold_pred_probs, axis=1)
        fold_acc = accuracy_score(y_test, fold_pred_classes)

        fold_predictions.append(fold_pred_probs)
        fold_test_accuracies.append(fold_acc)

        print(f"  Fold {fold_idx} test accuracy: {fold_acc:.4f}")

    ensemble_predictions = np.mean(fold_predictions, axis=0)
    ensemble_pred_classes = np.argmax(ensemble_predictions, axis=1)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred_classes)

    print(f"\nIndividual model test accuracies:")
    print(f"  Mean: {np.mean(fold_test_accuracies):.4f} +/- {np.std(fold_test_accuracies):.4f}")
    print(f"\nEnsemble test accuracy: {ensemble_accuracy:.4f}")
    print(f"Improvement over mean: {ensemble_accuracy - np.mean(fold_test_accuracies):+.4f}")

    test_predictions = ensemble_predictions
    test_pred_classes = ensemble_pred_classes
    test_accuracy = ensemble_accuracy

# Classification report
print("\nEnsemble Classification Report:")
classification_dict = classification_report(
    y_test,
    test_pred_classes,
    target_names=label_encoder.classes_,
    zero_division=0,
    output_dict=True
)

df_classification_report = pd.DataFrame(classification_dict).T
individual_classes = df_classification_report.iloc[:-3]
individual_classes_sorted = individual_classes.sort_values('f1-score', ascending=False)

print("\nTop 10 performing classes:")
print(individual_classes_sorted.head(10)[['precision', 'recall', 'f1-score', 'support']].round(3))

print("\nBottom 10 performing classes:")
print(individual_classes_sorted.tail(10)[['precision', 'recall', 'f1-score', 'support']].round(3))

print("\n" + "=" * 50)
print("OVERALL PERFORMANCE SUMMARY")
print("=" * 50)
print(f"  Overall Accuracy: {classification_dict['accuracy']:.4f}")
print(f"  Macro-Average F1: {classification_dict['macro avg']['f1-score']:.4f}")
print(f"  Weighted-Average F1: {classification_dict['weighted avg']['f1-score']:.4f}")
print(f"  Macro-Average Precision: {classification_dict['macro avg']['precision']:.4f}")
print(f"  Macro-Average Recall: {classification_dict['macro avg']['recall']:.4f}")


# ============================================================================
# 13. ROC/AUC Analysis
# ============================================================================
print("\n" + "=" * 60)
print("13. ROC/AUC ANALYSIS")
print("=" * 60)

# Binarize labels
y_test_binarized = label_binarize(y_test, classes=range(n_classes))
test_predictions_prob = test_predictions

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Micro-average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), test_predictions_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ROC for each class
for i in range(n_classes):
    if np.sum(y_test_binarized[:, i]) > 0:
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], test_predictions_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Macro-average ROC curve
all_class_indices = [i for i in roc_auc.keys() if isinstance(i, int)]
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i in all_class_indices:
    interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
    interp_tpr[0] = 0.0
    all_tpr.append(interp_tpr)

mean_tpr = np.mean(all_tpr, axis=0)
mean_tpr[-1] = 1.0
fpr["macro"] = mean_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
print(f"Macro-average AUC: {roc_auc['macro']:.4f}")

simple_macro_auc = np.mean([roc_auc[i] for i in roc_auc.keys() if isinstance(i, int)])
print(f"Simple macro-average AUC: {simple_macro_auc:.4f}")

# Plot ROC curves
plt.figure(figsize=(12, 9))
tumor_type_names = label_encoder.classes_
colors = [get_tumor_color(tumor_type_names[i], i) for i in all_class_indices]

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
plt.title(f'ROC Curves - SNV+CNA Ensemble - {COMBINED_MODEL_NAME}', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'snv_cna_ensemble_roc_curves_{COMBINED_MODEL_NAME}.png'), dpi=600, bbox_inches='tight')
plt.show()


# ============================================================================
# 14. Precision-Recall Analysis
# ============================================================================
print("\n" + "=" * 60)
print("14. PRECISION-RECALL ANALYSIS")
print("=" * 60)

precision = dict()
recall = dict()
average_precision = dict()

# Micro-average
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_binarized.ravel(),
                                                                test_predictions_prob.ravel())
average_precision["micro"] = average_precision_score(y_test_binarized, test_predictions_prob, average="micro")

# Per-class
for i in range(n_classes):
    if np.sum(y_test_binarized[:, i]) > 0:
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], test_predictions_prob[:, i])
        average_precision[i] = average_precision_score(y_test_binarized[:, i], test_predictions_prob[:, i])

# Macro-average
all_precision = []
recall_levels = np.linspace(0, 1, 100)

for i in all_class_indices:
    interp_precision = np.interp(recall_levels[::-1], recall[i][::-1], precision[i][::-1])[::-1]
    all_precision.append(interp_precision)

mean_precision = np.mean(all_precision, axis=0)
precision["macro"] = mean_precision
recall["macro"] = recall_levels
average_precision["macro"] = np.mean([average_precision[i] for i in average_precision.keys() if isinstance(i, int)])

print(f"Micro-average AP: {average_precision['micro']:.4f}")
print(f"Macro-average AP: {average_precision['macro']:.4f}")

# Plot precision-recall curves
plt.figure(figsize=(12, 9))
colors = [get_tumor_color(tumor_type_names[i], i) for i in all_class_indices]

for i, color in zip(all_class_indices, colors):
    class_name = tumor_type_names[i]
    plt.plot(recall[i], precision[i], color=color, lw=1.5, alpha=0.8,
             label=f'{class_name} (AP = {average_precision[i]:.3f})')

plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', linewidth=4,
         label=f'Micro-average (AP = {average_precision["micro"]:.3f})')
plt.plot(recall["macro"], precision["macro"], color='navy', linestyle=':', linewidth=4,
         label=f'Macro-average (AP = {average_precision["macro"]:.3f})')

baseline_precision = np.sum(y_test_binarized, axis=0) / len(y_test_binarized)
plt.axhline(y=baseline_precision.mean(), color='k', linestyle='--', lw=2, alpha=0.8,
            label=f'Random baseline (AP = {baseline_precision.mean():.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Precision-Recall Curves - SNV+CNA Ensemble - {COMBINED_MODEL_NAME}', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'snv_cna_ensemble_precision_recall_{COMBINED_MODEL_NAME}.png'), dpi=600, bbox_inches='tight')
plt.show()


# ============================================================================
# 15. Confusion Matrix
# ============================================================================
print("\n" + "=" * 60)
print("15. CONFUSION MATRIX")
print("=" * 60)

cm_all = confusion_matrix(y_test, test_pred_classes)
tumor_type_names = label_encoder.classes_

# Normalized confusion matrix
cm_normalized = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)
cm_normalized *= 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized,
            annot=True,
            fmt='.0f',
            cmap='Blues',
            xticklabels=tumor_type_names,
            yticklabels=tumor_type_names,
            cbar_kws={'label': 'Normalized Frequency (%)'},
            annot_kws={"size": 8})

plt.title(f'Confusion Matrix - SNV+CNA Ensemble (Normalized) - {COMBINED_MODEL_NAME}', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'snv_cna_ensemble_confusion_matrix_normalized_{COMBINED_MODEL_NAME}.png'), dpi=300, bbox_inches='tight')

# Raw counts
plt.figure(figsize=(10, 8))
sns.heatmap(cm_all,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=tumor_type_names,
            yticklabels=tumor_type_names,
            cbar_kws={'label': 'Count'})

plt.title(f'Confusion Matrix - SNV+CNA Ensemble (Raw Counts) - {COMBINED_MODEL_NAME}', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'snv_cna_ensemble_confusion_matrix_counts_{COMBINED_MODEL_NAME}.png'), dpi=300, bbox_inches='tight')
plt.show()

# Most common misclassifications
print("\nMost Common Misclassifications:")
misclass_pairs = []
for i in range(len(tumor_type_names)):
    for j in range(len(tumor_type_names)):
        if i != j and cm_all[i, j] > 0:
            misclass_pairs.append((cm_all[i, j], tumor_type_names[i], tumor_type_names[j]))

misclass_pairs.sort(reverse=True, key=lambda x: x[0])
for count, true_type, pred_type in misclass_pairs[:20]:
    print(f"{count:3d} cases: {true_type} -> {pred_type}")


# ============================================================================
# 16. Save Results
# ============================================================================
print("\n" + "=" * 60)
print("16. SAVING RESULTS")
print("=" * 60)

suffix = '_shuffled' if SHUFFLE_LABELS else ''

# Save label encoder
label_encoder_path = os.path.join(models_dir, f'snv_cna_ensemble_label_encoder_{COMBINED_MODEL_NAME}{suffix}.pkl')
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved: {label_encoder_path}")

# Save comprehensive results
results = {
    'snv_model_name': SNV_MODEL_NAME,
    'cna_model_name': CNA_MODEL_NAME,
    'combined_model_name': COMBINED_MODEL_NAME,
    'pca_applied': APPLY_PCA,
    'pca_n_components': pca_n_components_saved,
    'ensemble_performance': {
        'evaluation_mode': 'macro_nested_cv' if USE_MACRO_NESTED_CV else ('out_of_fold' if USE_FULL_DATASET_CV else 'test_set'),
        'ensemble_test_accuracy': test_accuracy,
        'mean_fold_test_accuracy': np.mean(fold_test_accuracies),
        'std_fold_test_accuracy': np.std(fold_test_accuracies),
        'fold_test_accuracies': fold_test_accuracies,
        'micro_auc': roc_auc['micro'],
        'macro_auc': roc_auc['macro'],
        'simple_macro_auc': simple_macro_auc,
        'micro_ap': average_precision['micro'],
        'macro_ap': average_precision['macro']
    },
    'cross_validation': {
        'n_macro_folds': N_MACRO_FOLDS if USE_MACRO_NESTED_CV else None,
        'n_inner_folds': N_INNER_FOLDS if USE_MACRO_NESTED_CV else N_FOLDS,
        'fold_val_accuracies': fold_val_accuracies,
        'fold_val_losses': fold_val_losses,
        'mean_val_accuracy': np.mean(fold_val_accuracies),
        'std_val_accuracy': np.std(fold_val_accuracies)
    },
    'roc_data': {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    },
    'precision_recall_data': {
        'precision': precision,
        'recall': recall,
        'average_precision': average_precision
    },
    'confusion_matrix': {
        'raw': cm_all,
        'normalized': cm_normalized,
        'class_names': tumor_type_names
    },
    'predictions': {
        'y_test': y_test,
        'y_pred_ensemble': test_pred_classes,
        'y_prob_ensemble': test_predictions,
        'fold_predictions': fold_predictions,
        'fold_test_accuracies': fold_test_accuracies
    },
    'training_histories': fold_histories,
    'feature_stats': {
        'n_snv_features': n_snv_features,
        'n_cna_features': n_cna_features,
        'n_combined_features': n_combined_features,
        'n_samples': n_samples_total,
        'n_train_samples': X_train.shape[0],
        'n_test_samples': 0 if X_test is None else X_test.shape[0]
    },
    'sample_info': {
        'sample_ids': sample_ids_list_final
    },
    'config': {
        'snv_model_name': SNV_MODEL_NAME,
        'cna_model_name': CNA_MODEL_NAME,
        'combined_model_name': COMBINED_MODEL_NAME,
        'pca_applied': APPLY_PCA,
        'n_macro_folds': N_MACRO_FOLDS,
        'n_inner_folds': N_INNER_FOLDS,
        'n_folds': N_FOLDS,
        'random_state': RANDOM_STATE,
        'shuffle_labels': SHUFFLE_LABELS,
        'use_macro_nested_cv': USE_MACRO_NESTED_CV,
        'use_full_dataset_cv': USE_FULL_DATASET_CV,
        'test_size': test_size if not USE_FULL_DATASET_CV else None,
        'min_samples_per_type': MIN_SAMPLES_PER_TYPE,
        'use_class_weights': USE_CLASS_WEIGHTS
    }
}

results_path = os.path.join(models_dir, f'snv_cna_ensemble_results_{COMBINED_MODEL_NAME}{suffix}.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Results saved: {results_path}")


print("\n" + "=" * 80)
print("SNV+CNA COMBINED TUMOR TYPE CLASSIFICATION COMPLETE!")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  SNV Model: {SNV_MODEL_NAME}")
print(f"  CNA Model: {CNA_MODEL_NAME}")
print(f"  Combined: {COMBINED_MODEL_NAME}")
if USE_MACRO_NESTED_CV:
    print(f"  Evaluation mode: Macro-fold nested CV ({N_MACRO_FOLDS} macro-folds x {N_INNER_FOLDS} inner folds)")
elif USE_FULL_DATASET_CV:
    print(f"  Evaluation mode: Full dataset CV (Out-of-Fold)")
else:
    print(f"  Evaluation mode: Separate test set ({test_size:.1%} held-out)")
if APPLY_PCA:
    print(f"  PCA components: {pca_n_components_saved}")

print(f"\nCross-Validation Performance:")
print(f"  Mean Validation Accuracy: {np.mean(fold_val_accuracies):.4f} +/- {np.std(fold_val_accuracies):.4f}")

if USE_MACRO_NESTED_CV:
    print(f"\nMacro-Fold Nested CV Performance:")
    print(f"  Individual Macro-Fold Accuracies: {[f'{acc:.4f}' for acc in macro_fold_accuracies]}")
    print(f"  Mean Macro-Fold Accuracy: {np.mean(macro_fold_accuracies):.4f} +/- {np.std(macro_fold_accuracies):.4f}")
    print(f"  Full Dataset Accuracy: {test_accuracy:.4f}")
elif USE_FULL_DATASET_CV:
    print(f"\nOut-of-Fold Performance:")
    print(f"  OOF Accuracy: {test_accuracy:.4f}")
else:
    print(f"\nTest Set Performance:")
    print(f"  Individual Models Mean: {np.mean(fold_test_accuracies):.4f} +/- {np.std(fold_test_accuracies):.4f}")
    print(f"  Ensemble Accuracy: {test_accuracy:.4f}")

print(f"\nMetrics:")
print(f"  Macro AUC: {roc_auc['macro']:.4f}")
print(f"  Micro AUC: {roc_auc['micro']:.4f}")
print(f"  Macro AP: {average_precision['macro']:.4f}")
print(f"  Micro AP: {average_precision['micro']:.4f}")
print(f"\nDataset:")
print(f"  Total samples: {n_samples_total:,}")
print(f"  SNV features: {n_snv_features}")
print(f"  CNA features: {n_cna_features}")
print(f"  Combined features: {n_combined_features}")
print(f"  Tumor Types: {n_classes}")
print("=" * 80)
