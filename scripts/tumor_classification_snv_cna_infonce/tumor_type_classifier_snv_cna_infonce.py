"""SNV+CNA InfoNCE-aligned Tumor Type Classifier (Figure 4 b-c).

Trains an ensemble of MLP tumor-type classifiers on per-sample features
extracted from the joint InfoNCE-aligned TESSERA model in
``scripts/tcga_pancan_snv_cna/``. The default features path points at
the LOH variant -- the headline classifier reported in Figure 4 b-c
(macro-AUC 0.987, micro-AUC 0.991, macro-AP 0.893, micro-AP 0.908). To
train the NoLOH ablation alongside, override ``FEATURES_PATH`` and
``OUTPUT_TAG`` (see README).

Features per sample:
  - SNV: weighted-average + max pooling of variant-level features +
    log1p(mutational_burden)
  - CNA: mean + max pooling of segment-level features +
    log1p(segment_count)
  - Concatenated to a single sample-level vector before the MLP.

Default training mode is macro-fold nested CV (5 outer / 10 inner)
so each sample is tested exactly once.

Usage
-----
    python tumor_type_classifier_snv_cna_infonce.py    # LOH variant (manuscript)

    FEATURES_PATH=../tcga_pancan_snv_cna/multimodal_features/TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH_multimodal_features.pkl \\
    OUTPUT_TAG=_noloh \\
        python tumor_type_classifier_snv_cna_infonce.py
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
# Default points at the LOH variant of the joint InfoNCE-aligned model
# (the manuscript headline classifier reported in Figure 4 b-c). To
# train the NoLOH variant ablation, set
#   FEATURES_PATH=../tcga_pancan_snv_cna/multimodal_features/TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH_multimodal_features.pkl
#   OUTPUT_TAG=_noloh
# which writes to models_macro_noloh/ and plots_noloh/ instead.
FEATURES_PATH = os.environ.get(
    "FEATURES_PATH",
    "../tcga_pancan_snv_cna/multimodal_features/"
    "TCGA_SNV_CNA_InfoNCE_per_sample_loss_multimodal_features.pkl",
)
OUTPUT_TAG = os.environ.get("OUTPUT_TAG", "")
MODELS_DIR = os.environ.get("MODELS_DIR", f"models_macro{OUTPUT_TAG}")
PLOTS_DIR = os.environ.get("PLOTS_DIR", f"plots{OUTPUT_TAG}")

print(f"Using features path: {FEATURES_PATH}")
print(f"Output: {MODELS_DIR}/ and {PLOTS_DIR}/")

# Training Configuration
SHUFFLE_LABELS = os.environ.get("SHUFFLE_LABELS", "0") != "0"
N_FOLDS = int(os.environ.get("N_FOLDS", "10"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))
USE_FULL_DATASET_CV = os.environ.get("USE_FULL_DATASET_CV", "0") != "0"
USE_MACRO_NESTED_CV = os.environ.get("USE_MACRO_NESTED_CV", "1") != "0"
test_size = float(os.environ.get("TEST_SIZE", "0.25"))
N_MACRO_FOLDS = int(os.environ.get("N_MACRO_FOLDS", "5"))
N_INNER_FOLDS = int(os.environ.get("N_INNER_FOLDS", "10"))

USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "0") != "0"
APPLY_PCA = os.environ.get("APPLY_PCA", "0") != "0"
PCA_N_COMPONENTS = (int(os.environ["PCA_N_COMPONENTS"])
                    if os.environ.get("PCA_N_COMPONENTS") else None)
PCA_EXPLAINED_VARIANCE_THRESHOLD = float(
    os.environ.get("PCA_EXPLAINED_VARIANCE_THRESHOLD", "0.95"))
MIN_SAMPLES_PER_TYPE = int(os.environ.get("MIN_SAMPLES_PER_TYPE", "100"))
TOP_N_TUMOR_TYPES = (int(os.environ["TOP_N_TUMOR_TYPES"])
                     if os.environ.get("TOP_N_TUMOR_TYPES") else None)

# Create plots and models directories
plots_dir = PLOTS_DIR
os.makedirs(plots_dir, exist_ok=True)

models_dir = MODELS_DIR
os.makedirs(models_dir, exist_ok=True)

print("=" * 80)
print("SNV+CNA INFONCE TUMOR TYPE CLASSIFIER - ENSEMBLE")
print("=" * 80)

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

print(f"Loading features from: {FEATURES_PATH}")
with open(FEATURES_PATH, 'rb') as f:
    data_out = pickle.load(f)

variant_features_train = data_out['train_variant_features']
variant_features_valid = data_out['valid_variant_features']

# Load variant data with sample IDs and tumor types
variant_train_data = data_out['train_data_snv']
variant_valid_data = data_out['valid_data_snv']

# Combine train and valid
variant_features = np.vstack([variant_features_train, variant_features_valid])
variant_data = pd.concat([variant_train_data, variant_valid_data], axis=0).reset_index(drop=True)

print(f"Loaded {variant_features.shape[0]:,} variant features")
print(f"  Train: {variant_features_train.shape[0]:,}")
print(f"  Valid: {variant_features_valid.shape[0]:,}")
print(f"  Feature dimensionality: {variant_features.shape[1]}")

# Clean up train/valid splits immediately after combining
print("  Cleaning up train/valid splits...")
del variant_features_train, variant_features_valid, variant_train_data, variant_valid_data
gc.collect()


# ============================================================================
# 2. Load CNA Features
# ============================================================================
print("\n" + "=" * 60)
print("2. LOADING CNA FEATURES")
print("=" * 60)

cna_features_train = data_out['train_cna_features']
cna_features_valid = data_out['valid_cna_features']
cna_train_data = data_out['train_data_cna']
cna_valid_data = data_out['valid_data_cna']

# Combine train and valid
cna_features = np.vstack([cna_features_train, cna_features_valid])
cna_data = pd.concat([cna_train_data, cna_valid_data], axis=0).reset_index(drop=True)

print(f"Loaded {cna_features.shape[0]:,} CNA features")
print(f"  Train: {cna_features_train.shape[0]:,}")
print(f"  Valid: {cna_features_valid.shape[0]:,}")
print(f"  Feature dimensionality: {cna_features.shape[1]}")

# Clean up train/valid splits immediately after combining
print("  Cleaning up train/valid splits...")
del cna_features_train, cna_features_valid, cna_train_data, cna_valid_data
gc.collect()


# ============================================================================
# 3. Filter for Valid Tumor Types
# ============================================================================
print("\n" + "=" * 60)
print("3. FILTERING TUMOR TYPES")
print("=" * 60)

# Combine COAD+READ → COADREAD and ESCA+STAD → ESCASTAD BEFORE counting
variant_data['type'] = variant_data['type'].replace({
    'COAD': 'COADREAD', 'READ': 'COADREAD',
    'ESCA': 'ESCASTAD', 'STAD': 'ESCASTAD',
})

# Get tumor type counts from variant data
data_tumor_type = variant_data[['Tumor_Sample_Barcode', 'type']].drop_duplicates()
data_tumor_type_counts = data_tumor_type['type'].value_counts()

# data_tumor_type = pd.read_csv('../../data/TCGA_PanCan/ncit.csv')
# data_tumor_type = data_tumor_type[['bcr_sample_barcode','ncit_label']].drop_duplicates()
# data_tumor_type.rename(columns={'bcr_sample_barcode':'Tumor_Sample_Barcode','ncit_label':'type'}, inplace=True)
# data_tumor_type_counts = data_tumor_type['type'].value_counts()

if TOP_N_TUMOR_TYPES is not None:
    valid_tumor_types = data_tumor_type_counts.nlargest(TOP_N_TUMOR_TYPES).index.tolist()
elif MIN_SAMPLES_PER_TYPE is not None:
    valid_tumor_types = data_tumor_type_counts[data_tumor_type_counts >= MIN_SAMPLES_PER_TYPE].index.tolist()
else:
    valid_tumor_types = data_tumor_type_counts.index.tolist()

tumor_type_dict = dict(zip(data_tumor_type['Tumor_Sample_Barcode'], data_tumor_type['type']))
variant_data['type'] = variant_data['Tumor_Sample_Barcode'].map(tumor_type_dict)

# Filter variant data
idx_keep_variant = variant_data['type'].isin(valid_tumor_types)
variant_data_filtered = variant_data[idx_keep_variant].reset_index(drop=True)
variant_features_filtered = variant_features[idx_keep_variant.values]

# Note: COAD and READ were already combined into COADREAD before filtering (see section 3)

# Extract first 15 characters from variant sample IDs to match CNA format
# Variant IDs: TCGA-02-0003-01A-01D-1490-08 (28 chars)
# CNA IDs:     TCGA-02-0003-01 (15 chars)
variant_data_filtered['Sample_ID_Short'] = variant_data_filtered['Tumor_Sample_Barcode'].str[:15]

print(f"Filtered to {len(valid_tumor_types)} tumor types")
print(f"Variant features after filtering: {variant_features_filtered.shape[0]:,}")


# ============================================================================
# 4. Aggregate SNV Features per Sample
# ============================================================================
print("\n" + "=" * 60)
print("4. AGGREGATING SNV FEATURES PER SAMPLE")
print("=" * 60)

print("\nUsing weighted average + max pooling aggregation...")

# Scale variant features before aggregation
scaler_variant = RobustScaler()
variant_features_scaled = scaler_variant.fit_transform(variant_features_filtered)

# Create DataFrame for aggregation using truncated sample IDs
variant_df = pd.DataFrame(variant_features_scaled)
variant_df.index = variant_data_filtered['Sample_ID_Short']  # Use truncated 15-char IDs
variant_df['vaf'] = 1.0  # Equal weighting (or use variant_data_filtered['vaf'] for VAF weighting)

def weighted_average(group):
    """Compute weighted average of features per sample"""
    group_features = group.drop(columns=['vaf']).values
    vaf_weights = group['vaf'].values

    if len(vaf_weights) == group_features.shape[0]:
        weighted_avg = np.average(group_features, axis=0, weights=vaf_weights)
        return pd.Series(weighted_avg)
    else:
        raise ValueError("Mismatch between group size and VAF weights.")

def max_pooling(group):
    """Compute max pooling of features per sample"""
    group_features = group.drop(columns=['vaf']).values
    max_features = np.max(group_features, axis=0)
    return pd.Series(max_features)

# Compute weighted average features
variant_agg_mean = variant_df.groupby(variant_df.index).apply(weighted_average)

# Compute max pooled features
variant_agg_max = variant_df.groupby(variant_df.index).apply(max_pooling)

# Concatenate mean and max features
variant_agg = pd.concat([variant_agg_mean, variant_agg_max], axis=1)
variant_agg.columns = [f'mean_{i}' for i in range(variant_agg_mean.shape[1])] + \
                      [f'max_{i}' for i in range(variant_agg_max.shape[1])]

# Add Total Mutational Burden as a feature (count of variants per sample)
mutational_burden = variant_df.groupby(variant_df.index).size()
variant_agg['mutational_burden'] = np.log1p(mutational_burden)

print(f"Aggregated SNV features: {variant_agg.shape[0]:,} samples")
print(f"  Feature dimensionality: {variant_agg.shape[1]} (mean + max pooling + mutational burden)")
print(f"    - Mean-pooled features: {variant_agg_mean.shape[1]}")
print(f"    - Max-pooled features: {variant_agg_max.shape[1]}")
print(f"  Mean mutational burden: {mutational_burden.mean():.1f} variants/sample")
print(f"  Median mutational burden: {mutational_burden.median():.1f} variants/sample")
print(f"  Range: {mutational_burden.min()}-{mutational_burden.max()} variants/sample")

# Clean up intermediate variant aggregation variables
print("  Cleaning up SNV aggregation intermediates...")
del variant_features_scaled, variant_df, variant_agg_mean, variant_agg_max, mutational_burden
gc.collect()


# ============================================================================
# 5. Aggregate CNA Features per Sample
# ============================================================================
print("\n" + "=" * 60)
print("5. AGGREGATING CNA FEATURES PER SAMPLE")
print("=" * 60)

print("\nUsing mean + max pooling aggregation for CNA features...")

# Scale CNA features before aggregation
scaler_cna = RobustScaler()
cna_features_scaled = scaler_cna.fit_transform(cna_features)

# Create DataFrame for aggregation (use Sample_16 column for sample ID)
cna_df = pd.DataFrame(cna_features_scaled)
cna_df.index = cna_data['Tumor_Sample_Barcode']

# Aggregate CNA features per sample (mean + max pooling)
cna_agg_mean = cna_df.groupby(cna_df.index).mean()
cna_agg_max = cna_df.groupby(cna_df.index).max()

# Concatenate mean and max features
cna_agg = pd.concat([cna_agg_mean, cna_agg_max], axis=1)
cna_agg.columns = [f'mean_{i}' for i in range(cna_agg_mean.shape[1])] + \
                  [f'max_{i}' for i in range(cna_agg_max.shape[1])]

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

# Clean up intermediate CNA aggregation variables
print("  Cleaning up CNA aggregation intermediates...")
del cna_features_scaled, cna_df, cna_agg_mean, cna_agg_max, cna_segment_count
gc.collect()


# ============================================================================
# 6. Find Samples with Both Variant and CNA Features
# ============================================================================
print("\n" + "=" * 60)
print("6. FINDING SAMPLES WITH BOTH MODALITIES")
print("=" * 60)

# Get sample IDs that have both variant and CNA features
variant_sample_ids = set(variant_agg.index)
cna_sample_ids = set(cna_agg.index)
common_sample_ids = variant_sample_ids & cna_sample_ids

print(f"Samples with variant features: {len(variant_sample_ids):,}")
print(f"Samples with CNA features: {len(cna_sample_ids):,}")
print(f"Samples with BOTH modalities: {len(common_sample_ids):,}")

# Filter to only samples with both modalities
common_sample_ids_list = sorted(list(common_sample_ids))
# df_soh_samples = pd.read_csv('soh_samples.csv')
# df_soh_samples['0'] = df_soh_samples['0'].astype(str)+'-01'
# common_sample_ids_list2 = list(df_soh_samples['0'])
# common_sample_ids_list = [x for x in common_sample_ids_list if x in common_sample_ids_list2]
variant_agg_common = variant_agg.loc[common_sample_ids_list]
cna_agg_common = cna_agg.loc[common_sample_ids_list]

print(f"\nFiltered samples:")
print(f"  Variant features: {variant_agg_common.shape}")
print(f"  CNA features: {cna_agg_common.shape}")


# ============================================================================
# 7. Concatenate Features
# ============================================================================
print("\n" + "=" * 60)
print("7. CONCATENATING MULTI-MODAL FEATURES")
print("=" * 60)

# Concatenate variant and CNA features
multimodal_features = np.hstack([
    variant_agg_common.values,
    cna_agg_common.values
])

# multimodal_features = cna_agg_common.values

print(f"Multi-modal feature matrix: {multimodal_features.shape}")
print(f"  Variant features: {variant_agg_common.shape[1]} (mean + max pooled)")
print(f"  CNA features: {cna_agg_common.shape[1]} (mean + max pooled)")
print(f"  Total features: {multimodal_features.shape[1]}")


# ============================================================================
# 7b. Optional PCA on Concatenated Features
# ============================================================================
if APPLY_PCA:
    print("\n" + "=" * 60)
    print("7b. APPLYING PCA TO CONCATENATED FEATURES")
    print("=" * 60)

    print(f"\nOriginal dimensionality: {multimodal_features.shape[1]}")

    # Standardize features before PCA
    scaler_pca = StandardScaler()
    multimodal_features_scaled = scaler_pca.fit_transform(multimodal_features)

    if PCA_N_COMPONENTS is None:
        # Use explained variance threshold to determine number of components
        print(f"Using explained variance threshold: {PCA_EXPLAINED_VARIANCE_THRESHOLD}")

        # Fit PCA with all components first to determine optimal n_components
        pca_full = PCA(random_state=RANDOM_STATE)
        pca_full.fit(multimodal_features_scaled)

        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= PCA_EXPLAINED_VARIANCE_THRESHOLD) + 1

        print(f"Selected {n_components} components (explains {cumsum_var[n_components-1]:.2%} of variance)")

        # Apply PCA with selected number of components
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        multimodal_features = pca.fit_transform(multimodal_features_scaled)

    else:
        n_components = PCA_N_COMPONENTS
        print(f"Using fixed {n_components} PCA components")

        # Fit and transform in one step
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        multimodal_features = pca.fit_transform(multimodal_features_scaled)

        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"Explained variance: {explained_var:.2%}")

    print(f"PCA applied: {multimodal_features.shape[1]} dimensions")
    print(f"  First 10 components explain: {np.sum(pca.explained_variance_ratio_[:10]):.2%}")
    print(f"  First 50 components explain: {np.sum(pca.explained_variance_ratio_[:min(50, n_components)]):.2%}")


# ============================================================================
# 8. Prepare Labels
# ============================================================================
print("\n" + "=" * 60)
print("8. PREPARING LABELS")
print("=" * 60)

# Get tumor types for common samples using truncated IDs
tumor_type_map = variant_data_filtered[['Sample_ID_Short', 'type']].drop_duplicates()
tumor_type_map = tumor_type_map.set_index('Sample_ID_Short')['type']

# Get labels for common samples
y = tumor_type_map.loc[common_sample_ids_list].values

# Remove any samples with missing labels
valid_label_mask = ~pd.isna(y)
multimodal_features = multimodal_features[valid_label_mask]
y = y[valid_label_mask]
sample_ids_final = np.array(common_sample_ids_list)[valid_label_mask]

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

# Final standardization of concatenated features
scaler_final = RobustScaler()
multimodal_features = scaler_final.fit_transform(multimodal_features)

print(f"\nTumor type distribution:")
tumor_type_counts = pd.Series(y).value_counts().sort_index()
for tumor_type, count in tumor_type_counts.head(10).items():
    print(f"  {tumor_type}: {count}")
if len(tumor_type_counts) > 10:
    print(f"  ... and {len(tumor_type_counts) - 10} more")


# ============================================================================
# 9. Train/Test Split (or use full dataset for CV)
# ============================================================================
print("\n" + "=" * 60)
print("9. SPLITTING DATA")
print("=" * 60)

if USE_MACRO_NESTED_CV:
    # Use macro-fold nested CV
    X_train = multimodal_features
    y_train = y_encoded
    X_test = None
    y_test = None

    print(f"Using macro-fold nested cross-validation")
    print(f"Total samples: {X_train.shape[0]:,}")
    print(f"Feature dimensionality: {X_train.shape[1]}")
    print(f"Evaluation will be across all samples (each tested exactly once)")

elif USE_FULL_DATASET_CV:
    # Use entire dataset for cross-validation
    X_train = multimodal_features
    y_train = y_encoded
    X_test = None
    y_test = None

    print(f"Using full dataset for cross-validation")
    print(f"Total samples: {X_train.shape[0]:,}")
    print(f"Feature dimensionality: {X_train.shape[1]}")
    print(f"Evaluation will be on out-of-fold predictions")
else:
    # Hold out a test set
    # Create indices to track which samples go to test set
    sample_indices = np.arange(len(multimodal_features))

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        multimodal_features, y_encoded, sample_indices,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Feature dimensionality: {X_train.shape[1]}")


# ============================================================================
# 10. Build and Train Ensemble with 5-Fold Cross-Validation
# ============================================================================
print("\n" + "=" * 60)
print("10. TRAINING ENSEMBLE WITH 5-FOLD CROSS-VALIDATION")
print("=" * 60)

def create_mlp_model(input_dim, n_classes, dropout_rate=0.5):
    """Create MLP classifier for multi-modal features"""
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
    skf_inner_base = N_INNER_FOLDS
    print(f"\nMACRO-FOLD NESTED CV: {N_MACRO_FOLDS} outer folds × {N_INNER_FOLDS} inner folds")
else:
    n_folds_outer = 1
    n_folds_inner = N_FOLDS
    skf_inner_base = N_FOLDS

skf = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=RANDOM_STATE)

# Store models and training histories
ensemble_models = []
fold_histories = []
fold_val_accuracies = []
fold_val_losses = []

# For macro-fold mode: store test set predictions and indices
if USE_MACRO_NESTED_CV:
    macro_fold_predictions_list = []  # Predictions for each macro-fold's test set
    macro_fold_test_indices_list = []  # Test indices for each macro-fold
    macro_fold_accuracies = []  # Accuracy for each macro-fold
else:
    # For full dataset CV mode: store out-of-fold predictions
    if USE_FULL_DATASET_CV:
        oof_predictions = np.zeros((len(X_train), n_classes))  # Out-of-fold predictions
        oof_indices = np.zeros(len(X_train), dtype=bool)  # Track which samples have predictions

if USE_MACRO_NESTED_CV:
    print(f"Total samples: {X_train.shape[0]:,}")
    print(f"Samples per macro-fold (test): {X_train.shape[0] // N_MACRO_FOLDS:,}")
    print(f"Samples per macro-fold (train): {X_train.shape[0] * (N_MACRO_FOLDS - 1) // N_MACRO_FOLDS:,}")
else:
    print(f"\nTraining {N_FOLDS} models using stratified {N_FOLDS}-fold cross-validation")
    print(f"Training set size: {X_train.shape[0]:,} samples")
print("")

# ============================================================================
# Clean up memory before training
# ============================================================================
print("Cleaning up memory before training...")

# Store feature stats before deletion
n_variant_features = variant_agg_common.shape[1]
n_cna_features = cna_agg_common.shape[1]
n_total_features_orig = multimodal_features.shape[1]
pca_n_components_saved = multimodal_features.shape[1] if APPLY_PCA else None
n_samples_total = len(y)
n_variant_only = len(variant_sample_ids - cna_sample_ids)
n_cna_only = len(cna_sample_ids - variant_sample_ids)
n_both = len(common_sample_ids)
sample_ids_list = list(sample_ids_final)

# Delete large intermediate variables
del variant_features
del cna_features
if 'variant_features_scaled' in locals():
    del variant_features_scaled
if 'cna_features_scaled' in locals():
    del cna_features_scaled
del variant_data
del variant_data_filtered
if 'variant_df' in locals():
    del variant_df
del variant_agg
del variant_agg_common
if 'variant_agg_mean' in locals():
    del variant_agg_mean
if 'variant_agg_max' in locals():
    del variant_agg_max
if 'variant_train_data' in locals():
    del variant_train_data
if 'variant_valid_data' in locals():
    del variant_valid_data
del cna_data
if 'cna_df' in locals():
    del cna_df
del cna_agg
del cna_agg_common
if 'cna_agg_mean' in locals():
    del cna_agg_mean
if 'cna_agg_max' in locals():
    del cna_agg_max
if 'cna_train_data' in locals():
    del cna_train_data
if 'cna_valid_data' in locals():
    del cna_valid_data
del scaler_variant
del scaler_cna
if 'scaler_pca' in locals():
    del scaler_pca
del scaler_final
del tumor_type_map
del common_sample_ids
del common_sample_ids_list
del variant_sample_ids
del cna_sample_ids
del multimodal_features
del y
del sample_ids_final
del data_out
if 'data_tumor_type' in locals():
    del data_tumor_type
if 'data_tumor_type_counts' in locals():
    del data_tumor_type_counts

gc.collect()
print("Memory cleanup complete!")

# ============================================================================
# Main training loop - either macro-fold nested or standard CV
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

            # Save model to macro folder
            suffix = '_shuffled' if SHUFFLE_LABELS else ''
            model_path = os.path.join(models_dir, f'macro{macro_idx}_inner{inner_idx}{suffix}.keras')
            model.save(model_path)

            # Store model
            macro_ensemble_models.append(model)
            fold_histories.append(history.history)
            ensemble_models.append(model)  # Also add to global list

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

        print(f"  Mean Inner CV Accuracy: {np.mean(macro_fold_val_accuracies):.4f} ± {np.std(macro_fold_val_accuracies):.4f}")
        print(f"  Ensemble Test Accuracy (Macro-Fold {macro_idx}): {macro_accuracy:.4f}")

        # Store results
        fold_val_accuracies.append(np.mean(macro_fold_val_accuracies))
        fold_val_losses.append(0)  # Not tracked for macro-fold
        macro_fold_predictions_list.append(ensemble_probs)
        macro_fold_test_indices_list.append(macro_test_idx)
        macro_fold_accuracies.append(macro_accuracy)

    print("\n" + "=" * 80)
    print("MACRO-FOLD NESTED CV COMPLETE")
    print("=" * 80)
    print(f"\nMacro-Fold Accuracies: {[f'{acc:.4f}' for acc in macro_fold_accuracies]}")
    print(f"Mean: {np.mean(macro_fold_accuracies):.4f} ± {np.std(macro_fold_accuracies):.4f}")

    # ========================================================================
    # Reconstruct full dataset predictions for evaluation
    # ========================================================================
    print("\nReconstructing full dataset predictions from macro-folds...")

    # Create array to hold all predictions in original order
    full_predictions = np.zeros((len(X_train), n_classes))
    full_labels = np.zeros(len(X_train), dtype=int)

    # Fill in predictions and labels from each macro-fold
    for macro_idx, (pred, indices) in enumerate(zip(macro_fold_predictions_list, macro_fold_test_indices_list)):
        full_predictions[indices] = pred
        full_labels[indices] = y_train[indices]

    # For compatibility with evaluation section
    test_predictions = full_predictions
    y_test = full_labels
    test_pred_classes = np.argmax(full_predictions, axis=1)
    test_accuracy = accuracy_score(full_labels, test_pred_classes)

    # Summary of macro-fold predictions
    print(f"Full dataset prediction accuracy: {test_accuracy:.4f}")
    print(f"Expected from macro-fold mean: {np.mean(macro_fold_accuracies):.4f}")
    print("")

else:
    # Standard CV (non-macro-fold)
    # Train one model per fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print("=" * 60)
        print(f"FOLD {fold_idx}/{N_FOLDS}")
        print("=" * 60)

        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        print(f"Fold {fold_idx} - Train: {X_fold_train.shape[0]:,} samples, Val: {X_fold_val.shape[0]:,} samples")

        # Compute class weights if enabled
        class_weights = None
        if USE_CLASS_WEIGHTS:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_fold_train),
                y=y_fold_train
            )
            # Convert to dictionary format expected by keras
            class_weights = {i: weight for i, weight in enumerate(class_weights)}
            print(f"  Using class weights (ratio of most common to least common class: {max(class_weights.values()) / min(class_weights.values()):.2f}x)")

        # Create and compile a fresh model for this fold
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

        # Set up callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=0
        )

        # Train model on this fold
        print(f"Training fold {fold_idx}...")
        history = model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=1000,
            batch_size=32,
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=1
        )

        # Evaluate on fold validation set
        val_loss, val_accuracy = model.evaluate(X_fold_val, y_fold_val, verbose=0)
        fold_val_accuracies.append(val_accuracy)
        fold_val_losses.append(val_loss)

        print(f"\nFold {fold_idx} Results:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")

        # Save model for this fold
        suffix = '_shuffled' if SHUFFLE_LABELS else ''
        model_path = f'tumor_type_multimodal_infonce_ensemble_fold{fold_idx}{suffix}.keras'
        model.save(model_path)
        print(f"  Model saved: {model_path}")

        # Store model and history
        ensemble_models.append(model)
        fold_histories.append(history.history)

        # If using full dataset CV, store out-of-fold predictions
        if USE_FULL_DATASET_CV:
            # Get predictions on the validation set for this fold
            val_predictions = model.predict(X_fold_val, verbose=0)
            oof_predictions[val_idx] = val_predictions
            oof_indices[val_idx] = True
            print(f"  Stored out-of-fold predictions for {len(val_idx)} samples")

        print("")

# Report ensemble training summary
print("=" * 60)
print("ENSEMBLE TRAINING COMPLETE")
print("=" * 60)
print(f"\nCross-Validation Results:")
print(f"  Mean Validation Accuracy: {np.mean(fold_val_accuracies):.4f} ± {np.std(fold_val_accuracies):.4f}")
print(f"  Mean Validation Loss: {np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}")
print(f"\nPer-Fold Validation Accuracies:")
for i, acc in enumerate(fold_val_accuracies, 1):
    print(f"  Fold {i}: {acc:.4f}")
print("")


# ============================================================================
# 11. Evaluate Ensemble (Test Set or Out-of-Fold Predictions or Macro-Fold)
# ============================================================================
print("\n" + "=" * 60)
if USE_MACRO_NESTED_CV:
    print("11. EVALUATING MACRO-FOLD NESTED CV RESULTS")
elif USE_FULL_DATASET_CV:
    print("11. EVALUATING ON OUT-OF-FOLD PREDICTIONS")
else:
    print("11. EVALUATING ENSEMBLE ON TEST SET")
print("=" * 60)

if USE_MACRO_NESTED_CV:
    # Macro-fold predictions already reconstructed and assigned to test_predictions, y_test, test_pred_classes, test_accuracy
    print(f"Macro-fold nested CV results:")
    print(f"  Full dataset accuracy: {test_accuracy:.4f}")
    print(f"  Individual macro-fold accuracies: {[f'{acc:.4f}' for acc in macro_fold_accuracies]}")

    # For compatibility with later code
    fold_predictions = None  # Not applicable in macro-fold mode
    fold_test_accuracies = macro_fold_accuracies

elif USE_FULL_DATASET_CV:
    # Use out-of-fold predictions for evaluation
    print(f"Using out-of-fold predictions for all {len(X_train):,} samples")

    # Verify all samples have out-of-fold predictions
    if not np.all(oof_indices):
        missing = np.sum(~oof_indices)
        print(f"WARNING: {missing} samples do not have out-of-fold predictions!")

    # Use out-of-fold predictions
    test_predictions = oof_predictions
    test_pred_classes = np.argmax(oof_predictions, axis=1)
    y_test = y_train
    test_accuracy = accuracy_score(y_train, test_pred_classes)

    print(f"\nOut-of-fold accuracy: {test_accuracy:.4f}")

    # For compatibility with later code, create fold_predictions and fold_test_accuracies
    fold_predictions = None  # Not applicable in OOF mode
    fold_test_accuracies = fold_val_accuracies  # Use validation accuracies instead

else:
    # Get predictions from each fold model on held-out test set
    print(f"Generating predictions from {len(ensemble_models)} models...")
    fold_predictions = []
    fold_test_accuracies = []

    for fold_idx, model in enumerate(ensemble_models, 1):
        # Get predictions from this fold's model
        fold_pred_probs = model.predict(X_test, verbose=0)
        fold_pred_classes = np.argmax(fold_pred_probs, axis=1)
        fold_acc = accuracy_score(y_test, fold_pred_classes)

        fold_predictions.append(fold_pred_probs)
        fold_test_accuracies.append(fold_acc)

        print(f"  Fold {fold_idx} test accuracy: {fold_acc:.4f}")

    # Average predictions across all folds (ensemble prediction)
    ensemble_predictions = np.mean(fold_predictions, axis=0)
    ensemble_pred_classes = np.argmax(ensemble_predictions, axis=1)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred_classes)

    print(f"\n" + "=" * 60)
    print(f"Individual model test accuracies:")
    print(f"  Mean: {np.mean(fold_test_accuracies):.4f} ± {np.std(fold_test_accuracies):.4f}")
    print(f"  Min:  {np.min(fold_test_accuracies):.4f}")
    print(f"  Max:  {np.max(fold_test_accuracies):.4f}")
    print(f"\nEnsemble test accuracy: {ensemble_accuracy:.4f}")
    print(f"Improvement over mean: {ensemble_accuracy - np.mean(fold_test_accuracies):+.4f}")
    print("=" * 60)

    # Use ensemble predictions for further analysis
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
summary_metrics = df_classification_report.iloc[-3:]

individual_classes_sorted = individual_classes.sort_values('f1-score', ascending=False)
print("\nTop 10 performing classes:")
print(individual_classes_sorted.head(10)[['precision', 'recall', 'f1-score', 'support']].round(3))

print("\nBottom 10 performing classes:")
print(individual_classes_sorted.tail(10)[['precision', 'recall', 'f1-score', 'support']].round(3))

print("\n" + "=" * 50)
print("OVERALL PERFORMANCE SUMMARY")
print("=" * 50)
print(f"• Overall Accuracy: {classification_dict['accuracy']:.4f}")
print(f"• Macro-Average F1: {classification_dict['macro avg']['f1-score']:.4f}")
print(f"• Weighted-Average F1: {classification_dict['weighted avg']['f1-score']:.4f}")
print(f"• Macro-Average Precision: {classification_dict['macro avg']['precision']:.4f}")
print(f"• Macro-Average Recall: {classification_dict['macro avg']['recall']:.4f}")


# ============================================================================
# 11b. Optimize Per-Class Thresholds for F1 Score
# ============================================================================
print("\n" + "=" * 60)
print("11b. OPTIMIZING PER-CLASS THRESHOLDS FOR F1 SCORE")
print("=" * 60)

# Binarize labels for threshold optimization
y_test_binarized = label_binarize(y_test, classes=range(n_classes))

# Store baseline metrics
baseline_accuracy = test_accuracy
baseline_f1_macro = classification_dict['macro avg']['f1-score']
baseline_f1_weighted = classification_dict['weighted avg']['f1-score']

print("\nBaseline (argmax) metrics:")
print(f"  Accuracy: {baseline_accuracy:.4f}")
print(f"  Macro F1: {baseline_f1_macro:.4f}")
print(f"  Weighted F1: {baseline_f1_weighted:.4f}")

# Find optimal thresholds for each class
print("\nComputing optimal thresholds for F1 maximization...")
optimal_thresholds = np.zeros(n_classes)
per_class_f1_improvement = []

for i in range(n_classes):
    if np.sum(y_test_binarized[:, i]) > 0:
        # Get precision-recall curve for this class
        precision_i, recall_i, thresholds_i = precision_recall_curve(
            y_test_binarized[:, i],
            test_predictions[:, i]
        )

        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-10)

        # Find threshold that maximizes F1
        best_idx = np.argmax(f1_scores)
        optimal_thresholds[i] = thresholds_i[best_idx]

        # Store improvement for tracking
        class_name = label_encoder.classes_[i]
        per_class_f1_improvement.append({
            'class': class_name,
            'threshold': optimal_thresholds[i],
            'max_f1': f1_scores[best_idx]
        })
    else:
        optimal_thresholds[i] = 0.5  # Default for classes with no positive samples

print(f"Computed optimal thresholds for all {n_classes} classes")

# Apply optimized thresholds to get new predictions
print("\nApplying optimized thresholds...")
test_pred_classes_optimized = np.zeros(len(test_predictions), dtype=int)

for sample_idx in range(len(test_predictions)):
    # Find class with highest probability that exceeds its threshold
    valid_classes = np.where(test_predictions[sample_idx] >= optimal_thresholds)[0]

    if len(valid_classes) > 0:
        # Pick class with highest probability among valid classes
        test_pred_classes_optimized[sample_idx] = valid_classes[np.argmax(test_predictions[sample_idx][valid_classes])]
    else:
        # If no class exceeds threshold, pick highest probability (fallback to argmax)
        test_pred_classes_optimized[sample_idx] = np.argmax(test_predictions[sample_idx])

# Re-evaluate with optimized thresholds
print("Re-evaluating metrics with optimized thresholds...")
test_accuracy_optimized = accuracy_score(y_test, test_pred_classes_optimized)

classification_dict_optimized = classification_report(
    y_test,
    test_pred_classes_optimized,
    target_names=label_encoder.classes_,
    zero_division=0,
    output_dict=True
)

optimized_f1_macro = classification_dict_optimized['macro avg']['f1-score']
optimized_f1_weighted = classification_dict_optimized['weighted avg']['f1-score']

# Show comparison
print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION RESULTS")
print("=" * 60)
print(f"\n{'Metric':<25} {'Argmax (baseline)':<20} {'Optimized':<20} {'Improvement':<15}")
print("-" * 80)
print(f"{'Accuracy':<25} {baseline_accuracy:<20.4f} {test_accuracy_optimized:<20.4f} {test_accuracy_optimized - baseline_accuracy:+.4f}")
print(f"{'Macro F1':<25} {baseline_f1_macro:<20.4f} {optimized_f1_macro:<20.4f} {optimized_f1_macro - baseline_f1_macro:+.4f}")
print(f"{'Weighted F1':<25} {baseline_f1_weighted:<20.4f} {optimized_f1_weighted:<20.4f} {optimized_f1_weighted - baseline_f1_weighted:+.4f}")

# Show per-class improvements (top 10 most improved)
print(f"\nTop improved classes:")
df_baseline = pd.DataFrame(classification_dict).T
df_optimized = pd.DataFrame(classification_dict_optimized).T

class_improvements = []
for i, class_name in enumerate(label_encoder.classes_):
    baseline_f1 = df_baseline.loc[class_name, 'f1-score'] if class_name in df_baseline.index else 0
    optimized_f1 = df_optimized.loc[class_name, 'f1-score'] if class_name in df_optimized.index else 0
    improvement = optimized_f1 - baseline_f1
    class_improvements.append((class_name, baseline_f1, optimized_f1, improvement, optimal_thresholds[i]))

class_improvements.sort(key=lambda x: abs(x[3]), reverse=True)

for idx, (class_name, baseline_f1, optimized_f1, improvement, threshold) in enumerate(class_improvements[:10]):
    print(f"  {idx+1}. {class_name:<10} F1: {baseline_f1:.4f} → {optimized_f1:.4f} ({improvement:+.4f})  Threshold: {threshold:.4f}")

# Use optimized predictions for remaining analysis
test_pred_classes = test_pred_classes_optimized
test_accuracy = test_accuracy_optimized
classification_dict = classification_dict_optimized

print(f"\nOptimized predictions will be used for ROC/AUC and remaining analyses.")


# ============================================================================
# 12. Ensemble ROC/AUC Analysis
# ============================================================================
print("\n" + "=" * 60)
print("12. ENSEMBLE ROC/AUC ANALYSIS")
print("=" * 60)

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

# Store for later use
all_class_indices_for_macro = all_class_indices

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
plt.title('ROC Curves - SNV+CNA InfoNCE Ensemble', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'multimodal_infonce_ensemble_roc_curves.png'), dpi=600, bbox_inches='tight')
plt.show()


# ============================================================================
# 13. Ensemble Precision-Recall Analysis
# ============================================================================
print("\n" + "=" * 60)
print("13. ENSEMBLE PRECISION-RECALL ANALYSIS")
print("=" * 60)

# Calculate precision-recall curves and average precision for each class
precision = dict()
recall = dict()
average_precision = dict()

# Calculate micro-average precision-recall
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_binarized.ravel(),
                                                                test_predictions_prob.ravel())
average_precision["micro"] = average_precision_score(y_test_binarized, test_predictions_prob,
                                                     average="micro")

# Calculate precision-recall for each class
for i in range(n_classes):
    if np.sum(y_test_binarized[:, i]) > 0:  # Only if class exists in test set
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i],
                                                            test_predictions_prob[:, i])
        average_precision[i] = average_precision_score(y_test_binarized[:, i],
                                                       test_predictions_prob[:, i])

# Calculate macro-average precision-recall curve
all_precision = []
recall_levels = np.linspace(0, 1, 100)

for i in all_class_indices_for_macro:
    # Interpolate precision at common recall levels
    interp_precision = np.interp(recall_levels[::-1], recall[i][::-1], precision[i][::-1])[::-1]
    all_precision.append(interp_precision)

# Calculate mean precision and macro-average AP
mean_precision = np.mean(all_precision, axis=0)
precision["macro"] = mean_precision
recall["macro"] = recall_levels
average_precision["macro"] = np.mean([average_precision[i] for i in average_precision.keys()
                                      if isinstance(i, int)])

# Plot precision-recall curves for all classes
plt.figure(figsize=(12, 9))

# Use fixed colors for consistency
colors = [get_tumor_color(tumor_type_names[i], i) for i in all_class_indices]

# Plot precision-recall curves for all classes
for i, color in zip(all_class_indices, colors):
    class_name = tumor_type_names[i]
    plt.plot(recall[i], precision[i], color=color, lw=1.5, alpha=0.8,
             label=f'{class_name} (AP = {average_precision[i]:.3f})')

# Plot micro-average precision-recall curve
plt.plot(recall["micro"], precision["micro"], color='deeppink', linestyle=':', linewidth=4,
         label=f'Micro-average (AP = {average_precision["micro"]:.3f})')

# Plot macro-average precision-recall curve
plt.plot(recall["macro"], precision["macro"], color='navy', linestyle=':', linewidth=4,
         label=f'Macro-average (AP = {average_precision["macro"]:.3f})')

# Add baseline (random classifier)
baseline_precision = np.sum(y_test_binarized, axis=0) / len(y_test_binarized)
plt.axhline(y=baseline_precision.mean(), color='k', linestyle='--', lw=2, alpha=0.8,
            label=f'Random baseline (AP = {baseline_precision.mean():.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - SNV+CNA InfoNCE Ensemble', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'multimodal_infonce_ensemble_precision_recall_curves.png'), dpi=600, bbox_inches='tight')
plt.show()

# Print Average Precision scores
print(f"\nMicro-average AP: {average_precision['micro']:.4f}")
print(f"Macro-average AP: {average_precision['macro']:.4f}")

print(f"\nTop 10 classes by Average Precision:")
class_aps = [(i, average_precision[i], label_encoder.classes_[i]) for i in average_precision.keys()
             if isinstance(i, int)]
class_aps.sort(key=lambda x: x[1], reverse=True)

for i, (class_idx, ap_score, class_name) in enumerate(class_aps[:10]):
    print(f"{i + 1:2d}. {class_name}: {ap_score:.4f}")

print(f"\nBottom 10 classes by Average Precision:")
for i, (class_idx, ap_score, class_name) in enumerate(class_aps[-10:]):
    print(f"{len(class_aps) - 9 + i:2d}. {class_name}: {ap_score:.4f}")


# ============================================================================
# 14. Ensemble Confusion Matrix
# ============================================================================
print("\n" + "=" * 60)
print("14. ENSEMBLE CONFUSION MATRIX")
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
            cbar_kws={'label': 'Normalized Frequency'},
            annot_kws={"size": 8})

plt.title('Confusion Matrix - InfoNCE Ensemble Multi-Modal (Normalized)', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'multimodal_infonce_ensemble_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')

# Raw counts
plt.figure(figsize=(10, 8))
sns.heatmap(cm_all,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=tumor_type_names,
            yticklabels=tumor_type_names,
            cbar_kws={'label': 'Count'})

plt.title('Confusion Matrix - InfoNCE Ensemble Multi-Modal (Raw Counts)', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'multimodal_infonce_ensemble_confusion_matrix_counts.png'), dpi=300, bbox_inches='tight')
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
    print(f"{count:3d} cases: {true_type} → {pred_type}")


# ============================================================================
# 14.5. Variant Count Performance Analysis
# ============================================================================
print("\n" + "=" * 60)
print("14.5. VARIANT COUNT PERFORMANCE ANALYSIS")
print("=" * 60)

# Only perform if we have evaluation data (standard CV with test set or macro-fold nested CV)
if (not USE_FULL_DATASET_CV and X_test is not None) or USE_MACRO_NESTED_CV:
    print("\nAnalyzing performance as a function of variants per sample...")

    # Load variant data to count variants per sample
    print("  Loading variant data for variant counts...")
    variant_train_data = pd.read_csv('../data/tcga/train_data_snv.csv')
    variant_valid_data = pd.read_csv('../data/tcga/valid_data_snv.csv')
    variant_data_all = pd.concat([variant_train_data, variant_valid_data], axis=0).reset_index(drop=True)

    # Count variants per sample (using 15-char sample ID for consistency)
    variant_data_all['Sample_ID_Short'] = variant_data_all['Tumor_Sample_Barcode'].str[:15]
    variant_counts = variant_data_all.groupby('Sample_ID_Short').size().reset_index(name='variant_count')
    variant_count_dict = dict(zip(variant_counts['Sample_ID_Short'], variant_counts['variant_count']))

    # Get variant counts for test set samples
    print("  Extracting variant counts for test set samples...")

    # Map sample IDs to variant counts
    test_variant_counts = []
    for i, sample_id in enumerate(sample_ids_list):
        if sample_id in variant_count_dict:
            test_variant_counts.append(variant_count_dict[sample_id])
        else:
            test_variant_counts.append(0)  # Default to 0 if not found

    test_variant_counts = np.array(test_variant_counts)

    # Extract variant counts for test set
    if USE_MACRO_NESTED_CV:
        # In macro-fold mode, y_test contains labels for all samples in full_labels
        # We need to get variant counts for all samples that were tested (which is all samples)
        test_sample_variant_counts = test_variant_counts
    else:
        # In standard mode, use the captured indices for the held-out test set
        test_sample_variant_counts = test_variant_counts[indices_test]

    # Compute test accuracy and confidence
    test_accuracy_per_sample = (test_pred_classes == y_test).astype(int)
    test_confidence_scores = np.max(test_predictions, axis=1)

    # Define bins
    bins = [0, 5, 10, 20, np.inf]
    bin_labels = ['0-5', '5-10', '10-20', '20+']

    bins = [0, 5, 10, 20, 100,500, np.inf]
    bin_labels = ['0-5', '5-10', '10-20', '20-100','100-500', '500+']

    # Assign samples to bins
    sample_bins = pd.cut(test_sample_variant_counts, bins=bins, labels=bin_labels, right=False)

    # Compute statistics per bin
    variant_count_stats = []
    for bin_label in bin_labels:
        mask = sample_bins == bin_label
        n_samples_bin = mask.sum()

        if n_samples_bin > 0:
            accuracy_bin = test_accuracy_per_sample[mask].mean()
            confidence_bin = test_confidence_scores[mask].mean()
            variant_count_bin = test_sample_variant_counts[mask]

            variant_count_stats.append({
                'bin': bin_label,
                'n_samples': n_samples_bin,
                'accuracy': accuracy_bin,
                'confidence': confidence_bin,
                'min_variants': variant_count_bin.min(),
                'max_variants': variant_count_bin.max(),
                'mean_variants': variant_count_bin.mean()
            })

    df_variant_stats = pd.DataFrame(variant_count_stats)

    print("\n  Variant Count Analysis Results:")
    print(df_variant_stats.to_string(index=False))

    # Save to CSV
    df_variant_stats.to_csv(f'ensemble_variant_count_performance.csv', index=False)
    print(f"\n  Saved: ensemble_variant_count_performance.csv")

    # Visualization 1: Accuracy by Variant Count Bins
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(range(len(df_variant_stats)), df_variant_stats['accuracy'],
                   color=plt.cm.viridis(np.linspace(0, 1, len(df_variant_stats))), alpha=0.7, edgecolor='black')
    plt.xlabel('Variant Count Bin', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Accuracy by Variant Count', fontsize=12, fontweight='bold')
    plt.xticks(range(len(df_variant_stats)), df_variant_stats['bin'])
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{df_variant_stats.iloc[i]["accuracy"]:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Visualization 2: Sample Count Distribution
    plt.subplot(2, 2, 2)
    bars2 = plt.bar(range(len(df_variant_stats)), df_variant_stats['n_samples'],
                   color=plt.cm.plasma(np.linspace(0, 1, len(df_variant_stats))), alpha=0.7, edgecolor='black')
    plt.xlabel('Variant Count Bin', fontsize=11)
    plt.ylabel('Number of Samples', fontsize=11)
    plt.title('Sample Distribution by Variant Count', fontsize=12, fontweight='bold')
    plt.xticks(range(len(df_variant_stats)), df_variant_stats['bin'])
    plt.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(df_variant_stats.iloc[i]["n_samples"])}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Visualization 3: Confidence Score by Variant Count
    plt.subplot(2, 2, 3)
    bars3 = plt.bar(range(len(df_variant_stats)), df_variant_stats['confidence'],
                   color=plt.cm.coolwarm(np.linspace(0, 1, len(df_variant_stats))), alpha=0.7, edgecolor='black')
    plt.xlabel('Variant Count Bin', fontsize=11)
    plt.ylabel('Mean Confidence Score', fontsize=11)
    plt.title('Prediction Confidence by Variant Count', fontsize=12, fontweight='bold')
    plt.xticks(range(len(df_variant_stats)), df_variant_stats['bin'])
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{df_variant_stats.iloc[i]["confidence"]:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Visualization 4: Scatter plot - Confidence vs Accuracy per bin
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(df_variant_stats['confidence'], df_variant_stats['accuracy'],
                        s=df_variant_stats['n_samples']*5, alpha=0.6, edgecolors='black', linewidth=2,
                        c=range(len(df_variant_stats)), cmap='tab10')
    for i, bin_label in enumerate(df_variant_stats['bin']):
        plt.annotate(bin_label,
                   (df_variant_stats.iloc[i]['confidence'], df_variant_stats.iloc[i]['accuracy']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    plt.xlabel('Mean Confidence Score', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Confidence vs Accuracy (bubble size = # samples)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'ensemble_variant_count_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved plot: plots/ensemble_variant_count_analysis.png")

    # Per-class analysis
    print("\n  Per-Class Performance by Variant Count:")
    print("  " + "-" * 80)

    per_class_stats = []
    for class_idx, class_name in enumerate(label_encoder.classes_):
        class_mask = y_test == class_idx
        if class_mask.sum() == 0:
            continue

        class_variant_counts = test_sample_variant_counts[class_mask]
        class_accuracy = test_accuracy_per_sample[class_mask]

        class_bins = pd.cut(class_variant_counts, bins=bins, labels=bin_labels, right=False)

        for bin_label in bin_labels:
            bin_mask = class_bins == bin_label
            if bin_mask.sum() > 0:
                bin_accuracy = class_accuracy[bin_mask].mean()
                n_in_bin = bin_mask.sum()
                print(f"  {class_name:8s} | {bin_label:6s}: {bin_accuracy:.3f} ({n_in_bin:2d} samples)")

                per_class_stats.append({
                    'tumor_type': class_name,
                    'variant_count_bin': bin_label,
                    'accuracy': bin_accuracy,
                    'n_samples': int(n_in_bin)
                })

    # Convert to dataframe and save
    df_per_class_stats = pd.DataFrame(per_class_stats)
    df_per_class_stats.to_csv(f'ensemble_per_class_variant_count_performance.csv', index=False)
    print(f"\n  Saved per-class stats: ensemble_per_class_variant_count_performance.csv")

    # Also create a pivot table for easier viewing
    pivot_per_class = df_per_class_stats.pivot_table(
        index='tumor_type',
        columns='variant_count_bin',
        values='accuracy',
        aggfunc='first'
    )
    pivot_per_class.to_csv(f'ensemble_per_class_variant_count_pivot.csv')
    print(f"  Saved pivot table: ensemble_per_class_variant_count_pivot.csv")

    print("\n  Per-Class Accuracy by Variant Count (Pivot Table):")
    print(pivot_per_class.round(3).to_string())

else:
    print("\nSkipping variant count analysis: Full dataset CV mode (no separate test set)")
    print("To enable this analysis, set USE_FULL_DATASET_CV=False")


# ============================================================================
# 15. Save Ensemble Results
# ============================================================================
print("\n" + "=" * 60)
print("15. SAVING ENSEMBLE RESULTS")
print("=" * 60)

suffix = '_shuffled' if SHUFFLE_LABELS else ''

# Save label encoder
label_encoder_path = os.path.join(models_dir, f'snv_cna_infonce_ensemble_label_encoder{suffix}.pkl')
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\nLabel encoder saved: {label_encoder_path}")

# Save individual fold models (already saved during training)
print(f"\nFold models saved:")
for fold_idx in range(1, N_FOLDS + 1):
    print(f"  - tumor_type_multimodal_infonce_ensemble_fold{fold_idx}{suffix}.keras")

# Save comprehensive ensemble results
results = {
    'model_type': 'snv_cna_infonce',
    'pca_applied': APPLY_PCA,
    'pca_n_components': pca_n_components_saved,
    'ensemble_performance': {
        'evaluation_mode': 'out_of_fold' if USE_FULL_DATASET_CV else 'test_set',
        'ensemble_test_accuracy': test_accuracy,  # This is either OOF accuracy or ensemble test accuracy
        'mean_fold_test_accuracy': np.mean(fold_test_accuracies),
        'std_fold_test_accuracy': np.std(fold_test_accuracies),
        'min_fold_test_accuracy': np.min(fold_test_accuracies),
        'max_fold_test_accuracy': np.max(fold_test_accuracies),
        'fold_test_accuracies': fold_test_accuracies,
        'micro_auc': roc_auc['micro'],
        'macro_auc': roc_auc['macro'],
        'simple_macro_auc': simple_macro_auc,
        'micro_ap': average_precision['micro'],
        'macro_ap': average_precision['macro']
    },
    'cross_validation': {
        'n_folds': N_FOLDS,
        'fold_val_accuracies': fold_val_accuracies,
        'fold_val_losses': fold_val_losses,
        'mean_val_accuracy': np.mean(fold_val_accuracies),
        'std_val_accuracy': np.std(fold_val_accuracies),
        'mean_val_loss': np.mean(fold_val_losses),
        'std_val_loss': np.std(fold_val_losses)
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
        'fold_predictions': fold_predictions,  # Individual fold predictions
        'fold_test_accuracies': fold_test_accuracies
    },
    'training_histories': fold_histories,
    'feature_stats': {
        'n_variant_features': n_variant_features,
        'n_cna_features': n_cna_features,
        'n_total_features': n_total_features_orig,
        'n_samples': n_samples_total,
        'n_train_samples': X_train.shape[0],
        'n_test_samples': 0 if X_test is None else X_test.shape[0]
    },
    'sample_info': {
        'common_sample_ids': sample_ids_list,
        'n_variant_only': n_variant_only,
        'n_cna_only': n_cna_only,
        'n_both': n_both
    },
    'config': {
        'model_type': 'snv_cna_infonce',
        'features_path': FEATURES_PATH,
        'pca_applied': APPLY_PCA,
        'n_folds': N_FOLDS,
        'random_state': RANDOM_STATE,
        'shuffle_labels': SHUFFLE_LABELS,
        'use_full_dataset_cv': USE_FULL_DATASET_CV,
        'test_size': test_size if not USE_FULL_DATASET_CV else None
    },
    'threshold_optimization': {
        'baseline_accuracy': baseline_accuracy,
        'baseline_f1_macro': baseline_f1_macro,
        'baseline_f1_weighted': baseline_f1_weighted,
        'optimized_accuracy': test_accuracy_optimized,
        'optimized_f1_macro': optimized_f1_macro,
        'optimized_f1_weighted': optimized_f1_weighted,
        'optimal_thresholds': optimal_thresholds.tolist(),
        'optimal_thresholds_per_class': {
            label_encoder.classes_[i]: float(optimal_thresholds[i])
            for i in range(n_classes)
        }
    }
}

results_path = os.path.join(models_dir, f'snv_cna_infonce_ensemble_results{suffix}.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f"Ensemble results saved: {results_path}")

print("\n" + "=" * 80)
print("SNV+CNA INFONCE TUMOR TYPE CLASSIFICATION COMPLETE!")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Model type: SNV+CNA InfoNCE (jointly trained)")
if USE_MACRO_NESTED_CV:
    print(f"  Evaluation mode: Macro-fold nested CV ({N_MACRO_FOLDS} macro-folds × {N_INNER_FOLDS} inner folds)")
elif USE_FULL_DATASET_CV:
    print(f"  Evaluation mode: Full dataset CV (Out-of-Fold)")
else:
    print(f"  Evaluation mode: Separate test set ({test_size:.1%} held-out)")
if APPLY_PCA:
    print(f"  PCA components: {pca_n_components_saved}")
print(f"\nCross-Validation Performance:")
print(f"  Mean Validation Accuracy: {np.mean(fold_val_accuracies):.4f} ± {np.std(fold_val_accuracies):.4f}")
print(f"  Mean Validation Loss: {np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}")

if USE_MACRO_NESTED_CV:
    print(f"\nMacro-Fold Nested CV Performance:")
    print(f"  Individual Macro-Fold Accuracies: {[f'{acc:.4f}' for acc in macro_fold_accuracies]}")
    print(f"  Mean Macro-Fold Accuracy: {np.mean(macro_fold_accuracies):.4f} ± {np.std(macro_fold_accuracies):.4f}")
    print(f"  Full Dataset Accuracy: {test_accuracy:.4f}")
elif USE_FULL_DATASET_CV:
    print(f"\nOut-of-Fold Performance:")
    print(f"  OOF Accuracy: {test_accuracy:.4f}")
else:
    print(f"\nTest Set Performance:")
    print(f"  Individual Models Mean: {np.mean(fold_test_accuracies):.4f} ± {np.std(fold_test_accuracies):.4f}")
    print(f"  Ensemble Accuracy: {test_accuracy:.4f}")
    print(f"  Improvement: {test_accuracy - np.mean(fold_test_accuracies):+.4f}")

print(f"\nMetrics:")
print(f"  Macro AUC: {roc_auc['macro']:.4f}")
print(f"  Micro AUC: {roc_auc['micro']:.4f}")
print(f"  Macro AP: {average_precision['macro']:.4f}")
print(f"  Micro AP: {average_precision['micro']:.4f}")
print(f"\nDataset:")
print(f"  Total samples: {n_samples_total:,}")
print(f"  Features: {n_total_features_orig} (mean+max pooled)")
print(f"    - Variant: {n_variant_features} (mean+max pooled)")
print(f"    - CNA: {n_cna_features} (mean+max pooled)")
print(f"  Tumor Types: {n_classes}")
print(f"  Ensemble Models: {N_FOLDS}")
print("=" * 80)
