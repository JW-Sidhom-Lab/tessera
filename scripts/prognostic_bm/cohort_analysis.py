"""
General-Purpose Cohort Analysis Script

Loads pretrained SNV + CNA variant features plus a user-provided clinical
metadata CSV, then performs:
  1. Per-sample feature aggregation (mean + max pooling)
  2. UMAP dimensionality reduction + GMM clustering
  3. Kaplan-Meier survival analysis by categorical variables and clusters
  4. Polynomial-Cox trajectory analysis on UMAP coordinates
  5. Risk stratification via continuous Cox risk scores

Must be run from inside `scripts/prognostic_bm/` because feature and data
paths are relative. See README.md for configuration details.

Features are loaded either from the raw pretrained pkls produced by
the foundation-model pretraining (``../tcga_pancan_snv/var_features/``
and ``../tcga_pancan_cna/cna_features/``) or, when ``FEATURES_SOURCE``
is set to ``infonce_per_sample_loss`` (the manuscript Figure 5
default), from the joint InfoNCE-aligned pkls in
``../tcga_pancan_snv_cna/multimodal_features/``. Sample metadata
(Tumor_Sample_Barcode, tumor type, etc.) comes from the train/valid
CSVs in ``../data/tcga/``. Clinical metadata (subtype, survival) comes
from ``../../data/prognostic/<cohort>/<cohort>_clinical_metadata.csv``.
"""

# ────────────────────────────────────────────────────────────────────────────
# DETERMINISM BLOCK -- must run BEFORE any numpy / sklearn / numba / umap import.
# UMAP, BLAS, and numba are multi-threaded by default; thread races cause the
# bit-level differences in UMAP coords across runs that propagate through the
# polynomial-Cox fit and produce different risk-group boundaries from one run
# to the next. Pinning everything to a single thread + seeding Python's hash
# and numpy's RNG eliminates this stochasticity.
# ────────────────────────────────────────────────────────────────────────────
import os as _os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    _os.environ[_v] = '1'
_os.environ['PYTHONHASHSEED'] = '0'

import random as _random
_random.seed(42)

import os
import gc
import pickle
import importlib
import platform
import pandas as pd
import numpy as np
np.random.seed(42)

from types import SimpleNamespace
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy import stats
import umap

import matplotlib
# Use interactive MacOSX backend on Mac, headless Agg elsewhere (e.g. Linux/headless)
matplotlib.use("MacOSX" if platform.system() == "Darwin" else "Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import distinctipy
import seaborn as sns

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

from feature_aggregation import aggregate_per_sample

# Optional: For time-dependent metrics
try:
    from sksurv.metrics import cumulative_dynamic_auc, brier_score as integrated_brier_score
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("WARNING: scikit-survival not installed. Time-dependent AUC and Brier scores will be skipped.")

# ============================================================================
# Cohort Selection
# ============================================================================
# Set COHORT to the name of a config file under configs/ (without .py),
# or any TCGA tumor type string (e.g. 'ACC', 'MESO').
# If no configs/<cohort>.py exists, _make_default_config() auto-derives
# the required fields from
# ../../data/prognostic/<cohort>/<cohort>_clinical_metadata.csv (if
# present) or from the pan-cancer ../../data/TCGA_PanCan/clinical.csv
# fallback.
#
# Hand-written configs ship for the manuscript Figure 5 cohorts:
# glioma, brca, prad. Any other TCGA type also works via the fallback.
COHORT = 'brca'

def _make_default_config(cohort_code: str) -> SimpleNamespace:
    """
    Build a default config namespace for any TCGA tumor type that has no
    hand-written configs/<cohort>.py.

    Discovery order:
      1. ../../data/prognostic/<cohort_lower>/<cohort_lower>_clinical_metadata.csv
         (cohort-specific, may contain subtype columns -> categorical
         analysis enabled)
      2. ../../data/TCGA_PanCan/clinical.csv filtered to
         type == cohort_upper (pan-cancer fallback; no subtype info ->
         survival-only mode)

    Survival column priority: DSS > OS > DSS_cr > PFI > DFI
    """
    _SURV_PAIRS = [
        ('DSS',    'DSS.time'),
        ('OS',     'OS.time'),
        ('DSS_cr', 'DSS.time.cr'),
        ('PFI',    'PFI.time'),
        ('DFI',    'DFI.time'),
    ]
    _SUBTYPE_COLS = [
        'Subtype', 'Histologic_Subtype', 'Molecular_Subtype', 'WHO2021', 'subtype',
    ]

    cohort_lower = cohort_code.lower()
    cohort_upper = cohort_code.upper()
    cohort_csv = f'../../data/prognostic/{cohort_lower}/{cohort_lower}_clinical_metadata.csv'
    pancancer_csv = '../../data/TCGA_PanCan/clinical.csv'

    ns = SimpleNamespace()
    ns._clinical_df = None   # used only for pan-cancer fallback

    if os.path.exists(cohort_csv):
        # ── Cohort-specific CSV ────────────────────────────────────────────────
        ns.CLINICAL_DATA_PATH = cohort_csv
        ns.PATIENT_ID_COL     = 'Patient_ID'
        header = pd.read_csv(cohort_csv, nrows=0)
        cols   = set(header.columns)

        # Survival columns
        for evt, tim in _SURV_PAIRS:
            if evt in cols and tim in cols:
                ns.SURV_EVENT_COL = evt
                ns.SURV_TIME_COL  = tim
                break
        else:
            raise ValueError(
                f"No recognised survival column pair found in {cohort_csv}.\n"
                f"Looked for: {_SURV_PAIRS}"
            )

        # Categorical subtype column (optional)
        ns.CATEGORICAL_VAR_COL = next((c for c in _SUBTYPE_COLS if c in cols), None)
        ns.CATEGORIES_INCLUDE  = None   # include all values when auto-detected

        print(f"  Auto-config [{cohort_upper}] from {cohort_csv}")

    elif os.path.exists(pancancer_csv):
        # ── Pan-cancer fallback ────────────────────────────────────────────────
        ns.CLINICAL_DATA_PATH = None  # signal: use _clinical_df instead
        ns.PATIENT_ID_COL     = 'Patient_ID'

        clin = pd.read_csv(pancancer_csv)
        clin_filt = clin[clin['type'] == cohort_upper].copy()
        if len(clin_filt) == 0:
            raise ValueError(
                f"No patients found for type='{cohort_upper}' in {pancancer_csv}.\n"
                f"Available types: {sorted(clin['type'].dropna().unique().tolist())}"
            )
        clin_filt = clin_filt.rename(columns={'bcr_patient_barcode': 'Patient_ID'})
        ns._clinical_df = clin_filt

        cols = set(clin_filt.columns)
        for evt, tim in _SURV_PAIRS:
            if evt in cols and tim in cols:
                ns.SURV_EVENT_COL = evt
                ns.SURV_TIME_COL  = tim
                break
        else:
            raise ValueError(
                f"No recognised survival column pair in pan-cancer CSV for {cohort_upper}.\n"
                f"Columns present: {sorted(cols)}"
            )

        ns.CATEGORICAL_VAR_COL = None
        ns.CATEGORIES_INCLUDE  = None

        print(f"  Auto-config [{cohort_upper}] from pan-cancer CSV "
              f"({len(clin_filt)} patients, survival-only mode)")

    else:
        raise FileNotFoundError(
            f"No config file or clinical data found for cohort '{cohort_code}'.\n"
            f"Tried (in order):\n"
            f"  configs/{cohort_lower}.py\n"
            f"  {cohort_csv}\n"
            f"  {pancancer_csv}"
        )

    # Shared required defaults (optional fields fall through to getattr defaults)
    ns.ADDITIONAL_CATEGORICAL_VARS = []
    ns.TIME_LIMIT = 1825
    ns.PCA_N_COMPONENTS = None   # skip PCA -- go straight to UMAP
    ns.SCALER_TOKENS_SNV = None  # match pan_cancer_prognostic_benchmark pipeline
    ns.SCALER_TOKENS_CNA = None
    ns.SCALER_AGGREGATED = None

    print(f"  Survival: event='{ns.SURV_EVENT_COL}', time='{ns.SURV_TIME_COL}'")
    if ns.CATEGORICAL_VAR_COL:
        print(f"  Categorical: '{ns.CATEGORICAL_VAR_COL}' (all values)")
    else:
        print(f"  Categorical: none (survival-only mode)")

    return ns


try:
    _cohort_cfg = importlib.import_module(f'configs.{COHORT}')
    print(f"Loading cohort config: configs/{COHORT}.py")
except ModuleNotFoundError:
    print(f"No config found for '{COHORT}' -- auto-detecting defaults ...")
    _cohort_cfg = _make_default_config(COHORT)

# ============================================================================
# Feature Source Selection
# ============================================================================
# FEATURES_SOURCE selects which pretrained backbone to use for SNV + CNA
# per-variant / per-segment features. Two manuscript-relevant options:
#   'raw'                     -- independently-pretrained global_25 SNV +
#                                attn_2 CNA features from tcga_pancan_snv/
#                                and tcga_pancan_cna/ (the concat baseline
#                                used in Figure 3 multimodal panels).
#   'infonce_per_sample_loss' -- joint InfoNCE-aligned features from
#                                tcga_pancan_snv_cna/ (the manuscript
#                                Figure 5 prognostic biomarker analyses).
# The OUTPUT_DIR is automatically suffixed with the source name (except
# for 'raw') so multiple runs don't collide.
FEATURES_SOURCE = 'infonce_per_sample_loss'

# ============================================================================
# Configuration
# ============================================================================

# ---- Data Paths (shared across cohorts) ----
SNV_FEATURES_DIR = '../tcga_pancan_snv/var_features'
CNA_FEATURES_DIR = '../tcga_pancan_cna/cna_features'
DATA_DIR = '../data/tcga'
SNV_MODEL_NAME = 'global_25'
CNA_MODEL_NAME = 'attn_2'

# Joint multimodal features (used when FEATURES_SOURCE != 'raw').
MULTIMODAL_FEATURES_DIR = '../tcga_pancan_snv_cna/multimodal_features'
_INFONCE_PKLS = {
    'infonce_per_sample_loss': 'TCGA_SNV_CNA_InfoNCE_per_sample_loss_multimodal_features.pkl',
}
_INFONCE_DIRS = {}  # all entries default to MULTIMODAL_FEATURES_DIR

# ---- Cohort-specific (from configs/<cohort>.py) ----
CLINICAL_DATA_PATH = _cohort_cfg.CLINICAL_DATA_PATH
CATEGORICAL_VAR_COL = _cohort_cfg.CATEGORICAL_VAR_COL
CATEGORIES_INCLUDE = _cohort_cfg.CATEGORIES_INCLUDE
SURV_TIME_COL = _cohort_cfg.SURV_TIME_COL
SURV_EVENT_COL = _cohort_cfg.SURV_EVENT_COL

# ---- Column Mappings (shared defaults, cohort may override) ----
PATIENT_ID_COL = getattr(_cohort_cfg, 'PATIENT_ID_COL', 'Patient_ID')
ADDITIONAL_CATEGORICAL_VARS = getattr(_cohort_cfg, 'ADDITIONAL_CATEGORICAL_VARS', [])

# ---- Survival Time Limit ----
# Right-censor follow-up at this many days. None to disable. 1825 = 5 years.
TIME_LIMIT = getattr(_cohort_cfg, 'TIME_LIMIT', 1825)

# ---- Survival Evaluation Timepoints ----
# Used for AUC(t) and Brier scores (requires scikit-survival).
# None to auto-generate [1yr, 2yr, 3yr, 5yr] up to TIME_LIMIT.
SURVIVAL_EVAL_TIMEPOINTS_CONFIG = getattr(
    _cohort_cfg, 'SURVIVAL_EVAL_TIMEPOINTS_CONFIG',
    [365, 730, 1095, 1461, 1800])
SURVIVAL_EVAL_TIMEPOINTS = None  # populated after loading survival data


# ---- UMAP Parameters ----
# ---- UMAP Parameters (cohort may override) ----
UMAP_N_NEIGHBORS = getattr(_cohort_cfg, 'UMAP_N_NEIGHBORS', 100)
UMAP_MIN_DIST = getattr(_cohort_cfg, 'UMAP_MIN_DIST', 0.5)
# Distance metric UMAP uses to build its neighborhood graph. 'euclidean' is the
# default; 'cosine' is often more principled for InfoNCE / contrastive-trained
# embeddings (directions are the discriminative signal, magnitudes are noise).
UMAP_METRIC = getattr(_cohort_cfg, 'UMAP_METRIC', 'cosine')
# Consensus UMAP: run N independent UMAP fits across different seeds and
# average the Procrustes-aligned embeddings. 1 = single run (legacy), >1 =
# robustness-oriented consensus. Absorbs seed-to-seed variation in local
# topology; does NOT guarantee bit-reproducibility across runs.
UMAP_N_CONSENSUS_SEEDS = getattr(_cohort_cfg, 'UMAP_N_CONSENSUS_SEEDS', 10)
# PCA preprocessing before UMAP. Float in (0,1) = fraction of variance to retain (e.g. 0.90);
# int = fixed number of components; None = skip PCA.
PCA_N_COMPONENTS = getattr(_cohort_cfg, 'PCA_N_COMPONENTS', None)
# Per-modality token-level scalers (SNV and CNA have different distribution shapes;
# legacy SCALER_TOKENS on the cohort config is respected as a shared fallback).
# None | 'robust' | 'standard'
_legacy_tok = getattr(_cohort_cfg, 'SCALER_TOKENS', 'robust')
SCALER_TOKENS_SNV = getattr(_cohort_cfg, 'SCALER_TOKENS_SNV', 'robust')
SCALER_TOKENS_CNA = getattr(_cohort_cfg, 'SCALER_TOKENS_CNA', 'robust')
# Scaler applied to aggregated sample-level vectors after pooling. None | 'robust' | 'standard'
SCALER_AGGREGATED = getattr(_cohort_cfg, 'SCALER_AGGREGATED', 'robust')
RANDOM_STATE = getattr(_cohort_cfg, 'RANDOM_STATE', 42)
PLOT_DPI = getattr(_cohort_cfg, 'PLOT_DPI', 300)
# True  = grey background dots for censored patients + colored dots for events
# False = events only, no censored dots shown
SURV_TIME_UMAP_SHOW_CENSORED = getattr(_cohort_cfg, 'SURV_TIME_UMAP_SHOW_CENSORED', True)

# ---- Clustering (cohort may override) ----
N_GMM_CLUSTERS = getattr(_cohort_cfg, 'N_GMM_CLUSTERS', 3)
CLUSTER_ON = getattr(_cohort_cfg, 'CLUSTER_ON', 'umap')  # 'umap' (2D) or 'pca' (99% variance)
GMM_COVARIANCE_TYPE = getattr(_cohort_cfg, 'GMM_COVARIANCE_TYPE', 'full')

# ---- Trajectory Analysis (cohort may override) ----
# POLYNOMIAL_DEGREE: fit Cox models from degree 1 to this, then pick by TRAJECTORY_MODEL.
#   degree 1 (linear): UMAP_1, UMAP_2
#   degree 2 (quadratic): + UMAP_1^2, UMAP_2^2 (+ UMAP_1*UMAP_2 if INCLUDE_INTERACTION_TERMS)
#   degree 3 (cubic): + UMAP_1^3, UMAP_2^3 (+ cross terms if INCLUDE_INTERACTION_TERMS)
POLYNOMIAL_DEGREE = getattr(_cohort_cfg, 'POLYNOMIAL_DEGREE', 1)
INCLUDE_INTERACTION_TERMS = getattr(_cohort_cfg, 'INCLUDE_INTERACTION_TERMS', True)
# 'best' picks the degree with the highest C-index; or specify an int.
TRAJECTORY_MODEL = getattr(_cohort_cfg, 'TRAJECTORY_MODEL', 'best')

# ---- Risk Grouping (cohort may override) ----
N_RISK_GROUPS = getattr(_cohort_cfg, 'N_RISK_GROUPS', 3)
# 'threshold' = fixed risk-score thresholds (clinical interpretation).
# 'quantile'  = equal-size groups (better statistical power).
RISK_GROUP_METHOD = getattr(_cohort_cfg, 'RISK_GROUP_METHOD', 'threshold')
BASE_POWER = getattr(_cohort_cfg, 'BASE_POWER', 1.0)  # threshold spacing on log scale
CONTOUR_RESOLUTION = getattr(_cohort_cfg, 'CONTOUR_RESOLUTION', 50)

# ---- Cluster-Specific Trajectory Analysis (cohort may override) ----
# When True, prompts the user to pick a single GMM cluster for trajectory analysis.
CLUSTER_TRAJ = getattr(_cohort_cfg, 'CLUSTER_TRAJ', False)

# ---- Combined Plot Panel Selection (cohort may override) ----
COMBINED_UMAP_PANELS = getattr(
    _cohort_cfg, 'COMBINED_UMAP_PANELS',
    ['categorical', 'clusters', 'risk_groups'])
COMBINED_SURVIVAL_PANELS = getattr(
    _cohort_cfg, 'COMBINED_SURVIVAL_PANELS',
    ['categorical', 'clusters', 'risk_groups'])

# ---- Missing Survival Data Handling ----
MISSING_SURVIVAL_HANDLING = getattr(_cohort_cfg, 'MISSING_SURVIVAL_HANDLING', 'warn')

# ---- Output ----
OUTPUT_DIR = getattr(_cohort_cfg, 'OUTPUT_DIR', f'analysis_results_{COHORT}')
# Suffix the output directory with the feature source so multiple runs don't collide.
if FEATURES_SOURCE != 'raw':
    OUTPUT_DIR = f'{OUTPUT_DIR}_{FEATURES_SOURCE}'
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/umap_individual', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/survival_individual', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/concordance_plots/gmm_clusters/counts', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/concordance_plots/gmm_clusters/percentages', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/concordance_plots/risk_groups/counts', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/concordance_plots/risk_groups/percentages', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/umap_associations', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots/survival_metrics', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data/survival_metrics', exist_ok=True)

print("=" * 80)
print(f"COHORT ANALYSIS: {COHORT.upper()} -- {CATEGORICAL_VAR_COL} vs GMM Clusters")
print(f"Features: {FEATURES_SOURCE}")
print(f"Clinical: {CLINICAL_DATA_PATH}")
print(f"Survival: time_col='{SURV_TIME_COL}', event_col='{SURV_EVENT_COL}' (TIME_LIMIT={TIME_LIMIT})")
print(f"Output:   {OUTPUT_DIR}/")
print("=" * 80)

# ============================================================================
# Validate Configuration
# ============================================================================
if N_RISK_GROUPS < 2:
    raise ValueError(f"N_RISK_GROUPS must be >= 2 (got {N_RISK_GROUPS})")
if BASE_POWER <= 0:
    raise ValueError(f"BASE_POWER must be > 0 (got {BASE_POWER}). Use 1.0 for linear, 1.5+ to exaggerate extremes")
print(f"\n✓ Configuration valid (N_RISK_GROUPS={N_RISK_GROUPS}, BASE_POWER={BASE_POWER})")

# ============================================================================
# 1. Load Features
# ============================================================================
print("\n1. LOADING FEATURES")
print("-" * 80)

if FEATURES_SOURCE == 'raw':
    # --- Raw pretrained features (default) ---
    # SNV: raw pretrained features + per-variant metadata CSV
    snv_features_path = os.path.join(SNV_FEATURES_DIR, f'TCGA_PanCan_SNV_{SNV_MODEL_NAME}_features.pkl')
    print(f"Loading SNV features from: {snv_features_path}")
    with open(snv_features_path, 'rb') as f:
        snv_out = pickle.load(f)

    snv_train = pd.read_csv(os.path.join(DATA_DIR, 'train_data_snv.csv'))
    snv_valid = pd.read_csv(os.path.join(DATA_DIR, 'valid_data_snv.csv'))
    variant_features = np.vstack([snv_out['train'], snv_out['valid']])
    variant_data = pd.concat([snv_train, snv_valid], axis=0).reset_index(drop=True)
    del snv_out, snv_train, snv_valid

    # CNA: pretrained pkl bundles both features and per-segment metadata
    cna_features_path = os.path.join(CNA_FEATURES_DIR, f'TCGA_PanCan_CNA_{CNA_MODEL_NAME}_cna_features.pkl')
    print(f"Loading CNA features from: {cna_features_path}")
    with open(cna_features_path, 'rb') as f:
        cna_out = pickle.load(f)

    cna_features = np.vstack([cna_out['train_features_cna'], cna_out['valid_features_cna']])
    cna_data = pd.concat([cna_out['train_data_cna'], cna_out['valid_data_cna']], axis=0).reset_index(drop=True)
    del cna_out
    gc.collect()

elif FEATURES_SOURCE in _INFONCE_PKLS:
    # --- Bundled InfoNCE multimodal pkl (old scripts workflow) ---
    # Single pkl carries both SNV + CNA features and their per-row metadata.
    _base_dir = _INFONCE_DIRS.get(FEATURES_SOURCE, MULTIMODAL_FEATURES_DIR)
    multimodal_path = os.path.join(_base_dir, _INFONCE_PKLS[FEATURES_SOURCE])
    print(f"Loading multimodal InfoNCE features from: {multimodal_path}")
    with open(multimodal_path, 'rb') as f:
        mm_out = pickle.load(f)

    variant_features = np.vstack([mm_out['train_variant_features'], mm_out['valid_variant_features']])
    variant_data = pd.concat([mm_out['train_data_snv'], mm_out['valid_data_snv']], axis=0).reset_index(drop=True)
    cna_features = np.vstack([mm_out['train_cna_features'], mm_out['valid_cna_features']])
    cna_data = pd.concat([mm_out['train_data_cna'], mm_out['valid_data_cna']], axis=0).reset_index(drop=True)
    del mm_out
    gc.collect()

else:
    raise ValueError(
        f"Unknown FEATURES_SOURCE='{FEATURES_SOURCE}'. "
        f"Expected one of: 'raw', {list(_INFONCE_PKLS.keys())}"
    )

print(f"Variant features: {variant_features.shape[0]:,} x {variant_features.shape[1]}")
print(f"CNA features: {cna_features.shape[0]:,} x {cna_features.shape[1]}")

# ============================================================================
# 2. Load Clinical Data
# ============================================================================
print("\n2. LOADING CLINICAL DATA")
print("-" * 80)

clinical = None
if hasattr(_cohort_cfg, '_clinical_df') and _cohort_cfg._clinical_df is not None:
    clinical = _cohort_cfg._clinical_df.copy()
    print(f"Clinical data (pan-cancer fallback, {COHORT.upper()}): {len(clinical)} patients")
elif CLINICAL_DATA_PATH is not None:
    clinical = pd.read_csv(CLINICAL_DATA_PATH)
    print(f"Clinical data: {len(clinical)} samples")
else:
    print("(Skipped - CLINICAL_DATA_PATH is None)")

# Extract patient ID from sample barcode (first 12 chars)
variant_data['Patient_ID'] = variant_data['Tumor_Sample_Barcode'].astype(str).str[:12]
cna_data['Patient_ID'] = cna_data['Tumor_Sample_Barcode'].astype(str).str[:12]

# Get unique patients
patients_all = set(variant_data['Patient_ID'].unique()) | set(cna_data['Patient_ID'].unique())
print(f"Unique patients in features: {len(patients_all)}")

# ============================================================================
# 3. Filter by Categorical Variable (if specified)
# ============================================================================
print("\n3. FILTERING BY CATEGORICAL VARIABLE")
print("-" * 80)

clinical_filtered = None
patients_keep = None

if clinical is not None and CATEGORICAL_VAR_COL is not None:
    # Merge clinical data with features to get categorical variable
    if CATEGORICAL_VAR_COL not in clinical.columns:
        raise ValueError(f"Column '{CATEGORICAL_VAR_COL}' not found in clinical data")

    # Get mapping of patient to categorical variable
    patient_to_category = dict(zip(clinical[PATIENT_ID_COL], clinical[CATEGORICAL_VAR_COL]))

    # Add categorical variable to patient set
    patients_with_category = set(clinical[PATIENT_ID_COL])
    print(f"Patients with {CATEGORICAL_VAR_COL}: {len(patients_with_category)}")
    print(f"\n{CATEGORICAL_VAR_COL} distribution:")
    for cat, count in clinical[CATEGORICAL_VAR_COL].value_counts().items():
        print(f"  {cat}: {count}")

    # Filter if CATEGORIES_INCLUDE specified
    if CATEGORIES_INCLUDE is not None:
        clinical_filtered = clinical[clinical[CATEGORICAL_VAR_COL].isin(CATEGORIES_INCLUDE)].copy()
        # Assign 'unknown' to categories not in list
        clinical_filtered[CATEGORICAL_VAR_COL] = clinical_filtered[CATEGORICAL_VAR_COL].where(
            clinical_filtered[CATEGORICAL_VAR_COL].isin(CATEGORIES_INCLUDE), 'unknown'
        )
        patients_keep = set(clinical_filtered[PATIENT_ID_COL])
        print(f"\nAfter filtering to {CATEGORIES_INCLUDE}: {len(patients_keep)} patients")
    else:
        # Keep all patients, use 'unknown' for missing values
        clinical_filtered = clinical.copy()
        clinical_filtered[CATEGORICAL_VAR_COL] = clinical_filtered[CATEGORICAL_VAR_COL].fillna('unknown')
        patients_keep = set(clinical_filtered[PATIENT_ID_COL])
elif clinical is not None and CATEGORICAL_VAR_COL is None:
    # No categorical variable but clinical file defines the patient cohort -- use it as a whitelist
    clinical_filtered = clinical.copy()
    patients_keep = set(clinical_filtered[PATIENT_ID_COL])
    print(f"No categorical variable -- using clinical file as patient whitelist ({len(patients_keep)} patients)")
else:
    print("(Skipped - clinical data or CATEGORICAL_VAR_COL is None)")

# ============================================================================
# 4. Filter Variants and CNA at Patient Level
# ============================================================================
print("\n4. FILTERING VARIANTS AND CNA")
print("-" * 80)

# Use patients_keep if clinical data exists and filtering was done, otherwise use all patients
if patients_keep is not None:
    var_filt_idx = variant_data['Patient_ID'].isin(patients_keep)
    cna_filt_idx = cna_data['Patient_ID'].isin(patients_keep)
else:
    # No filtering - use all patients
    var_filt_idx = pd.Series(True, index=variant_data.index)
    cna_filt_idx = pd.Series(True, index=cna_data.index)

var_data_filt = variant_data[var_filt_idx]
cna_data_filt = cna_data[cna_filt_idx]
var_feat_filt = variant_features[var_filt_idx]
cna_feat_filt = cna_features[cna_filt_idx]

print(f"Variant segments after filtering: {len(var_data_filt):,}")
print(f"CNA segments after filtering: {len(cna_data_filt):,}")

# ============================================================================
# 5. Aggregate Features
# ============================================================================
print("\n5. AGGREGATING FEATURES")
print("-" * 80)


def plot_concordance(y_var_name, y_var_data, x_var_name, x_var_data, output_path, percentage=False):
    """
    Create a concordance heatmap comparing two categorical variables.

    Args:
        y_var_name: Name of variable for y-axis (rows)
        y_var_data: Pandas Series with y-axis variable data
        x_var_name: Name of variable for x-axis (columns)
        x_var_data: Pandas Series with x-axis variable data
        output_path: Path to save the plot
        percentage: If True, show percentages (row-wise); if False, show counts
    """
    # Create contingency table with y_var as rows, x_var as columns
    contingency = pd.crosstab(y_var_data, x_var_data, margins=False)

    # Calculate data to display
    if percentage:
        # Calculate percentages for each row (row-wise percentages)
        display_data = contingency.div(contingency.sum(axis=1), axis=0) * 100
        fmt_str = '.1f'
        cbar_label = 'Percentage (%)'
    else:
        display_data = contingency
        fmt_str = 'd'
        cbar_label = 'Count'

    # Create figure with heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(display_data, annot=True, fmt=fmt_str, cmap='Blues',
                cbar_kws={'label': cbar_label}, ax=ax, linewidths=0.5, linecolor='gray')

    plot_type = "Percentages" if percentage else "Counts"
    ax.set_title(f'Concordance: {y_var_name} vs {x_var_name} ({plot_type})',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel(x_var_name, fontweight='bold')
    ax.set_ylabel(y_var_name, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_umap_association_metrics(data, umap_cols, categorical_var):
    """
    Calculate association metrics between UMAP coordinates (2D combined) and a categorical variable.

    Args:
        data: DataFrame containing UMAP columns and categorical variable
        umap_cols: List of UMAP column names ['UMAP_1', 'UMAP_2']
        categorical_var: Name of categorical variable column

    Returns:
        Dictionary with metrics: F_stat, P_value, R_squared, Silhouette_score, N_categories, Significant
    """
    # Remove missing values
    data_clean = data[[umap_cols[0], umap_cols[1], categorical_var]].dropna()

    if len(data_clean) < 2:
        return None

    categories = data_clean[categorical_var].unique()
    n_categories = len(categories)

    if n_categories < 2:
        return None  # Need at least 2 categories

    # Separate UMAP coordinates by category
    groups_umap1 = [data_clean[data_clean[categorical_var] == cat][umap_cols[0]].values for cat in categories]
    groups_umap2 = [data_clean[data_clean[categorical_var] == cat][umap_cols[1]].values for cat in categories]

    # ANOVA F-test for each UMAP axis
    f_stat_1, p_val_1 = stats.f_oneway(*groups_umap1)
    f_stat_2, p_val_2 = stats.f_oneway(*groups_umap2)

    # Average F-statistic and use minimum p-value (conservative)
    f_stat_avg = (f_stat_1 + f_stat_2) / 2
    p_val_conservative = min(p_val_1, p_val_2)

    # Calculate R-squared (effect size) for each axis
    # Total variance
    var_total_1 = np.var(data_clean[umap_cols[0]], ddof=1)
    var_total_2 = np.var(data_clean[umap_cols[1]], ddof=1)

    # Between-group variance for each axis
    grand_mean_1 = data_clean[umap_cols[0]].mean()
    grand_mean_2 = data_clean[umap_cols[1]].mean()

    var_between_1 = sum(len(data_clean[data_clean[categorical_var] == cat]) *
                        (data_clean[data_clean[categorical_var] == cat][umap_cols[0]].mean() - grand_mean_1)**2
                        for cat in categories) / (len(data_clean) - 1)
    var_between_2 = sum(len(data_clean[data_clean[categorical_var] == cat]) *
                        (data_clean[data_clean[categorical_var] == cat][umap_cols[1]].mean() - grand_mean_2)**2
                        for cat in categories) / (len(data_clean) - 1)

    r2_1 = var_between_1 / var_total_1 if var_total_1 > 0 else 0
    r2_2 = var_between_2 / var_total_2 if var_total_2 > 0 else 0
    r2_avg = (r2_1 + r2_2) / 2

    # Silhouette score (measure of cluster separation in 2D space)
    umap_2d = data_clean[[umap_cols[0], umap_cols[1]]].values
    labels = pd.factorize(data_clean[categorical_var])[0]

    try:
        silhouette = silhouette_score(umap_2d, labels)
    except Exception as e:
        silhouette = np.nan

    # Determine significance
    significant = p_val_conservative < 0.05

    return {
        'Variable': categorical_var,
        'N_Categories': n_categories,
        'F_Statistic': f_stat_avg,
        'P_Value': p_val_conservative,
        'R_Squared': r2_avg,
        'Silhouette_Score': silhouette,
        'Significant': significant
    }

def generate_polynomial_features(umap_data, degree, include_interaction=True, feature_names=['UMAP_1', 'UMAP_2']):
    """
    Generate polynomial features with separate control over pure powers and interaction terms.

    Args:
        umap_data: DataFrame with UMAP_1 and UMAP_2 columns (already scaled)
        degree: Maximum power for individual variables (1, 2, 3, ...)
        include_interaction: Whether to include cross-product terms (i>0 and j>0)
        feature_names: Names of the two input features

    Returns:
        DataFrame with polynomial features

    Examples:
        degree=1, interaction=False → UMAP_1, UMAP_2 (2 features)
        degree=1, interaction=True  → UMAP_1, UMAP_2, UMAP_1*UMAP_2 (3 features)
        degree=2, interaction=False → UMAP_1, UMAP_2, UMAP_1^2, UMAP_2^2 (4 features)
        degree=2, interaction=True  → UMAP_1, UMAP_2, UMAP_1^2, UMAP_1*UMAP_2, UMAP_2^2 (5 features)
        degree=3, interaction=False → UMAP_1, UMAP_2, UMAP_1^2, UMAP_2^2, UMAP_1^3, UMAP_2^3 (6 features)
        degree=3, interaction=True  → All above + UMAP_1^2*UMAP_2, UMAP_1*UMAP_2^2 (9 features)
    """
    poly_features = pd.DataFrame(index=umap_data.index)

    # Pure terms go up to `degree`; cross terms go up to max(degree, 2) so that
    # U1*U2 is always included when include_interaction=True, even at degree=1.
    cross_degree = max(degree, 2) if include_interaction else degree
    max_total = cross_degree

    for total_degree in range(1, max_total + 1):
        for i_power in range(total_degree + 1):
            j_power = total_degree - i_power

            # Separate pure terms (one variable only) from interaction terms (both variables)
            is_pure_term = (i_power == 0 or j_power == 0)
            is_interaction = (i_power > 0 and j_power > 0)

            # Skip pure terms beyond the requested degree
            if is_pure_term and (i_power > degree or j_power > degree):
                continue

            # Skip interaction terms if include_interaction=False
            if is_interaction and not include_interaction:
                continue

            # Create feature name
            if i_power == 0:
                feature_name = f'{feature_names[1]}^{j_power}' if j_power > 1 else feature_names[1]
            elif j_power == 0:
                feature_name = f'{feature_names[0]}^{i_power}' if i_power > 1 else feature_names[0]
            else:
                i_part = f'{feature_names[0]}^{i_power}' if i_power > 1 else feature_names[0]
                j_part = f'{feature_names[1]}^{j_power}' if j_power > 1 else feature_names[1]
                feature_name = f'{i_part}*{j_part}'

            # Compute feature value
            poly_features[feature_name] = (umap_data[feature_names[0]] ** i_power) * (umap_data[feature_names[1]] ** j_power)

    return poly_features

def compute_survival_cindex(surv_data, grouping_var_name, surv_time_col, surv_event_col, debug=False):
    """
    Compute concordance index (C-index) for a survival grouping variable.

    BUG FIX (2025): Changed to use continuous risk scores instead of categorical Cox C-index.
    This ensures fair comparison between categorical and continuous models.

    Args:
        surv_data: DataFrame with survival data
        grouping_var_name: Name of categorical grouping variable
        surv_time_col: Name of survival time column
        surv_event_col: Name of event column
        debug: If True, print detailed debugging information

    Returns:
        float: Concordance index (0-1) computed from continuous risk scores, or NaN if computation fails
    """
    try:
        # Remove rows with missing data in the required columns
        data_clean = surv_data[[grouping_var_name, surv_time_col, surv_event_col]].dropna()

        if debug:
            print(f"\n    === C-INDEX DEBUG for '{grouping_var_name}' ===")
            print(f"    Original cohort size: {len(surv_data)}")
            print(f"    After dropna: {len(data_clean)}")
            print(f"    Unique groups: {sorted(data_clean[grouping_var_name].unique())}")
            print(f"    Group counts: {data_clean[grouping_var_name].value_counts().to_dict()}")
            print(f"    Patients: {list(data_clean.index[:10])}..." if len(data_clean) > 10 else f"    Patients: {list(data_clean.index)}")

        if len(data_clean) < 3:
            return np.nan

        # Create dummy variables for categorical variable
        dummy_vars = pd.get_dummies(data_clean[grouping_var_name], drop_first=True, prefix='group')

        # Prepare data for Cox model
        cox_data = data_clean[[surv_time_col, surv_event_col]].copy()
        for col in dummy_vars.columns:
            cox_data[col] = dummy_vars[col].values

        # Fit Cox model
        if len(dummy_vars.columns) > 0:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col=surv_time_col, event_col=surv_event_col)

            # Compute risk scores
            risk_scores = cph.predict_partial_hazard(cox_data.drop(columns=[surv_time_col, surv_event_col]))

            if debug:
                print(f"    Dummy variables: {list(dummy_vars.columns)}")
                print(f"    Risk scores - min: {risk_scores.min():.4f}, max: {risk_scores.max():.4f}, mean: {risk_scores.mean():.4f}, std: {risk_scores.std():.4f}")
                print(f"    C-index from Cox model: {cph.concordance_index_:.4f}")
                print(f"    Cox coefficients:")
                for idx, row in cph.summary.iterrows():
                    print(f"      {str(idx):25s}: coef={row['coef']:7.4f}, HR={np.exp(row['coef']):6.3f}, p={row['p']:.4f}")
                print(f"    === END DEBUG ===\n")

            # Return the built-in C-index from the fitted Cox model
            # (Using concordance_index() directly can introduce sign/alignment bugs)
            return cph.concordance_index_
        else:
            return np.nan
    except Exception as e:
        print(f"    WARNING: Error computing C-index: {e}")
        return np.nan

def compute_comprehensive_survival_metrics(surv_data, grouping_var_name, surv_time_col,
                                           surv_event_col, timepoints, cox_model=None,
                                           risk_scores=None):
    """
    Compute comprehensive survival metrics for a grouping variable.

    Args:
        surv_data: DataFrame with survival data
        grouping_var_name: Name of categorical grouping variable
        surv_time_col: Name of survival time column
        surv_event_col: Name of event column
        timepoints: List of time values to evaluate survival at
        cox_model: Fitted CoxPHFitter for time-dependent metrics (optional)
        risk_scores: Risk scores from Cox model (optional)

    Returns:
        Dictionary with metrics:
        - c_index: Harrell's C-index
        - logrank_p: Log-rank test p-value
        - logrank_stat: Log-rank test statistic
        - median_survival: Dict of median survival per group
        - survival_at_t: Dict of survival probabilities at timepoints
        - auc_at_t: Time-dependent AUC at timepoints (if Cox model provided)
        - brier_at_t: Brier scores at timepoints (if Cox model provided)
    """
    metrics = {}

    # Remove rows with missing data
    data_clean = surv_data[[grouping_var_name, surv_time_col, surv_event_col]].dropna()

    try:
        # 1. C-index
        metrics['c_index'] = compute_survival_cindex(surv_data, grouping_var_name,
                                                      surv_time_col, surv_event_col, debug=True)

        # 2. Log-rank test
        groups = sorted(data_clean[grouping_var_name].unique())
        if len(groups) >= 2:
            result = multivariate_logrank_test(data_clean[surv_time_col],
                                               data_clean[grouping_var_name],
                                               data_clean[surv_event_col])
            metrics['logrank_p'] = result.p_value
            metrics['logrank_stat'] = result.test_statistic
        else:
            metrics['logrank_p'] = np.nan
            metrics['logrank_stat'] = np.nan

        # 3. Median survival per group
        metrics['median_survival'] = {}
        for group in groups:
            mask = data_clean[grouping_var_name] == group
            if mask.sum() > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(data_clean.loc[mask, surv_time_col],
                        data_clean.loc[mask, surv_event_col])
                metrics['median_survival'][group] = kmf.median_survival_time_

        # 4. Survival at timepoints
        metrics['survival_at_t'] = {t: {} for t in timepoints}
        for group in groups:
            mask = data_clean[grouping_var_name] == group
            if mask.sum() > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(data_clean.loc[mask, surv_time_col],
                        data_clean.loc[mask, surv_event_col])
                for t in timepoints:
                    try:
                        metrics['survival_at_t'][t][group] = kmf.predict(t)
                    except:
                        metrics['survival_at_t'][t][group] = np.nan

        # 5. Time-dependent AUC(t) - only if Cox model and risk scores provided
        if cox_model is not None and risk_scores is not None and SKSURV_AVAILABLE:
            metrics['auc_at_t'] = {}
            metrics['auc_integrated'] = np.nan
            try:
                y_train = Surv.from_arrays(data_clean[surv_event_col].values.astype(bool),
                                            data_clean[surv_time_col].values)
                risk_scores_clean = risk_scores.loc[data_clean.index].values

                # Compute AUC at all timepoints and get integrated AUC
                try:
                    auc_scores, mean_auc = cumulative_dynamic_auc(y_train, y_train, risk_scores_clean, times=timepoints)
                    metrics['auc_integrated'] = mean_auc
                    for i, t in enumerate(timepoints):
                        metrics['auc_at_t'][t] = auc_scores[i]
                except Exception as batch_error:
                    # Fallback to computing individual timepoints if batch fails
                    valid_auc_timepoints = []
                    for t in timepoints:
                        try:
                            auc_t, _ = cumulative_dynamic_auc(y_train, y_train, risk_scores_clean, times=[t])
                            metrics['auc_at_t'][t] = auc_t[0]
                            valid_auc_timepoints.append(t)
                        except Exception as timepoint_error:
                            metrics['auc_at_t'][t] = np.nan

                    # Compute integrated AUC from valid timepoints
                    if valid_auc_timepoints:
                        valid_aucs = [metrics['auc_at_t'][t] for t in valid_auc_timepoints if not np.isnan(metrics['auc_at_t'][t])]
                        if valid_aucs:
                            metrics['auc_integrated'] = np.mean(valid_aucs)
            except Exception as e:
                metrics['auc_at_t'] = {t: np.nan for t in timepoints}
                metrics['auc_integrated'] = np.nan

    except Exception as e:
        print(f"    WARNING: Error computing comprehensive metrics: {e}")
        # Return partial metrics with NaN values
        if 'c_index' not in metrics:
            metrics['c_index'] = np.nan
        metrics['logrank_p'] = np.nan
        metrics['logrank_stat'] = np.nan
        metrics['median_survival'] = {}
        metrics['survival_at_t'] = {t: {} for t in timepoints}

    return metrics

# Aggregate using shared pipeline: (optional scaler) → mean+max pool → (optional scaler)
snv_X, snv_ids = aggregate_per_sample(var_feat_filt, var_data_filt, append_tmb=True, scale_tokens=SCALER_TOKENS_SNV, scale_aggregated=SCALER_AGGREGATED)
cna_X, cna_ids = aggregate_per_sample(cna_feat_filt, cna_data_filt, append_tmb=True, scale_tokens=SCALER_TOKENS_CNA, scale_aggregated=SCALER_AGGREGATED)

var_agg = pd.DataFrame(snv_X, index=snv_ids)
cna_agg = pd.DataFrame(cna_X, index=cna_ids)
print(f"Aggregated samples: {len(var_agg)}")

del var_feat_filt, cna_feat_filt
gc.collect()

# ============================================================================
# 6. Merge Features and Clinical Data
# ============================================================================
print("\n6. MERGING FEATURES WITH CLINICAL DATA")
print("-" * 80)

samples_both = set(var_agg.index) & set(cna_agg.index)
features = pd.concat([var_agg.loc[list(samples_both)], cna_agg.loc[list(samples_both)]], axis=1)

# Add sample to patient mapping
sample_to_patient = {s: p for s, p in zip(var_data_filt['Tumor_Sample_Barcode'], var_data_filt['Patient_ID'])}
features['Patient_ID'] = features.index.map(sample_to_patient)

# Merge with clinical data (include all columns for exploratory variables)
if clinical_filtered is not None:
    clinical_cols_to_merge = [PATIENT_ID_COL] + ([CATEGORICAL_VAR_COL] if CATEGORICAL_VAR_COL is not None else []) + ADDITIONAL_CATEGORICAL_VARS
    clinical_cols_to_merge = list(set(clinical_cols_to_merge))  # Remove duplicates
    clinical_cols_available = [col for col in clinical_cols_to_merge if col in clinical_filtered.columns]
    features = features.merge(clinical_filtered[clinical_cols_available],
                             left_on='Patient_ID', right_on=PATIENT_ID_COL, how='left')

# ============================================================================
# 7. Load Survival Data
# ============================================================================
print("\n7. LOADING SURVIVAL DATA")
print("-" * 80)

surv_time_col = None
surv_event_col = None

if clinical is not None:
    # Validate survival columns are specified
    if SURV_TIME_COL is None or SURV_EVENT_COL is None:
        raise ValueError(
            "SURV_TIME_COL and SURV_EVENT_COL must be explicitly specified.\n"
            "These should match the column names in your clinical CSV.\n"
            f"Current values: SURV_TIME_COL='{SURV_TIME_COL}', SURV_EVENT_COL='{SURV_EVENT_COL}'\n"
            "Example:\n"
            "  SURV_TIME_COL = 'OS.time'\n"
            "  SURV_EVENT_COL = 'OS'"
        )

    surv_time_col = SURV_TIME_COL
    surv_event_col = SURV_EVENT_COL
    print(f"Survival columns: time='{surv_time_col}', event='{surv_event_col}'")

    # Load survival columns from the clinical CSV
    if clinical_filtered is None:
        raise ValueError(
            "Clinical data is required for survival analysis but CLINICAL_DATA_PATH is None."
        )
    if surv_time_col not in clinical_filtered.columns or surv_event_col not in clinical_filtered.columns:
        raise ValueError(
            f"Survival columns '{surv_time_col}' and/or '{surv_event_col}' not found in "
            f"{CLINICAL_DATA_PATH}. Available columns: {list(clinical_filtered.columns)}"
        )
    surv_by_patient = clinical_filtered[[PATIENT_ID_COL, surv_time_col, surv_event_col]].set_index(PATIENT_ID_COL)
    print(f"Survival data loaded from clinical CSV ({CLINICAL_DATA_PATH})")

    # Map survival to features
    features[surv_time_col] = features['Patient_ID'].map(surv_by_patient[surv_time_col])
    features[surv_event_col] = features['Patient_ID'].map(surv_by_patient[surv_event_col])

    # Apply time limit censoring (if specified)
    if TIME_LIMIT is not None:
        print(f"\nApplying time limit censoring: {TIME_LIMIT} days")
        # Find samples beyond time limit
        beyond_limit = features[surv_time_col] > TIME_LIMIT
        n_censored = beyond_limit.sum()
        if n_censored > 0:
            # Censor these samples at TIME_LIMIT
            features.loc[beyond_limit, surv_time_col] = TIME_LIMIT
            features.loc[beyond_limit, surv_event_col] = 0  # Mark as censored (no event)
            print(f"  Censored {n_censored} samples at time limit")

    # Handle missing survival data
    n_total = len(features)
    n_with_surv = features[surv_time_col].notna().sum()
    n_missing = n_total - n_with_surv

    print(f"\nSurvival data status:")
    print(f"  Total samples: {n_total}")
    print(f"  With survival: {n_with_surv}")
    print(f"  Missing: {n_missing}")

    # Apply missing data handling policy
    if MISSING_SURVIVAL_HANDLING == 'strict' and n_missing > 0:
        raise ValueError(
            f"MISSING_SURVIVAL_HANDLING='strict' but {n_missing} samples have missing survival data.\n"
            f"Options:\n"
            f"  1. Set MISSING_SURVIVAL_HANDLING='warn' to drop missing samples with warning\n"
            f"  2. Set MISSING_SURVIVAL_HANDLING='drop' to silently drop missing samples\n"
            f"  3. Fix your clinical data to have complete survival data"
        )
    elif MISSING_SURVIVAL_HANDLING == 'warn' and n_missing > 0:
        print(f"\n⚠️  WARNING: Removing {n_missing} samples with missing survival data")
        features = features[features[surv_time_col].notna()].copy()
        print(f"  Proceeding with {len(features)} samples")
    elif MISSING_SURVIVAL_HANDLING == 'drop' and n_missing > 0:
        features = features[features[surv_time_col].notna()].copy()

    print(f"\nFinal dataset: {len(features)} samples")

    # Compute survival evaluation timepoints based on user configuration and TIME_LIMIT
    print("\n--- Determining AUC(t) and Brier Score Evaluation Timepoints ---")

    # Step 1: Determine candidate timepoints
    if SURVIVAL_EVAL_TIMEPOINTS_CONFIG is not None:
        # Use user-specified timepoints
        candidate_timepoints = list(SURVIVAL_EVAL_TIMEPOINTS_CONFIG)
        print(f"Using user-specified timepoints: {candidate_timepoints}")
    else:
        # Use standard clinical timepoints up to TIME_LIMIT or max observed time
        standard_times = [365, 730, 1095, 1825, 3650]  # 1yr, 2yr, 3yr, 5yr, 10yr
        if TIME_LIMIT is not None:
            candidate_timepoints = sorted([t for t in standard_times if t <= TIME_LIMIT])
            if TIME_LIMIT not in candidate_timepoints:
                candidate_timepoints.append(TIME_LIMIT)
            candidate_timepoints = sorted(candidate_timepoints)
            print(f"Auto-generated timepoints (up to TIME_LIMIT={TIME_LIMIT}d): {candidate_timepoints}")
        else:
            # Use max observed time if no limit
            max_time = features[surv_time_col].max()
            candidate_timepoints = sorted([t for t in standard_times if t <= max_time])
            if max_time not in candidate_timepoints:
                candidate_timepoints.append(max_time)
            candidate_timepoints = sorted(candidate_timepoints)
            print(f"Auto-generated timepoints (up to max observed={max_time:.0f}d): {candidate_timepoints}")

    # Use specified timepoints directly, no validation
    SURVIVAL_EVAL_TIMEPOINTS = candidate_timepoints
    print(f"Using evaluation timepoints: {SURVIVAL_EVAL_TIMEPOINTS}")
else:
    print("(Skipped - clinical data is None)")

# ============================================================================
# 8. UMAP and Clustering
# ============================================================================
print("\n8. UMAP AND CLUSTERING")
print("-" * 80)

# Extract features (exclude non-numeric columns and exploratory variables)
cols_to_exclude = {'Patient_ID', PATIENT_ID_COL, CATEGORICAL_VAR_COL, surv_time_col, surv_event_col, 'UMAP_1', 'UMAP_2', 'Cluster'}
cols_to_exclude.update(ADDITIONAL_CATEGORICAL_VARS)
feature_cols = [c for c in features.columns if c not in cols_to_exclude]
X = features[feature_cols].values.astype(np.float32)

# Scaling already handled inside aggregate_per_sample (SCALE_TOKENS / SCALE_AGGREGATED)
X_scaled = X

# Optional PCA before UMAP
if PCA_N_COMPONENTS is not None:
    pca_pre = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
    X_scaled = pca_pre.fit_transform(X_scaled)
    print(f"PCA preprocessing: {X.shape[1]} → {pca_pre.n_components_} components "
          f"({pca_pre.explained_variance_ratio_.sum()*100:.1f}% variance retained)")

# UMAP -- single run (UMAP_N_CONSENSUS_SEEDS == 1) or consensus of several
# runs across seeds, aligned via orthogonal Procrustes and averaged. Consensus
# smooths out seed-to-seed variation in local neighborhood topology while
# preserving the overall geometry (scale/orientation of the reference run).
if UMAP_N_CONSENSUS_SEEDS <= 1:
    umap_model = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                           metric=UMAP_METRIC,
                           random_state=RANDOM_STATE, n_components=2,
                           n_jobs=1, verbose=0)
    umap_coords = umap_model.fit_transform(X_scaled)
else:
    from scipy.linalg import orthogonal_procrustes
    print(f"Consensus UMAP: averaging {UMAP_N_CONSENSUS_SEEDS} Procrustes-aligned runs "
          f"(metric={UMAP_METRIC!r})")
    runs = []
    for i in range(UMAP_N_CONSENSUS_SEEDS):
        m = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                      metric=UMAP_METRIC,
                      random_state=RANDOM_STATE + i, n_components=2,
                      n_jobs=1, verbose=0)
        runs.append(m.fit_transform(X_scaled))
        print(f"  seed {RANDOM_STATE + i}: run {i + 1}/{UMAP_N_CONSENSUS_SEEDS}")
    ref = runs[0] - runs[0].mean(axis=0)
    aligned = [ref]
    for r in runs[1:]:
        rc = r - r.mean(axis=0)
        R, _ = orthogonal_procrustes(rc, ref)
        aligned.append(rc @ R)
    umap_coords = np.mean(aligned, axis=0)

features['UMAP_1'] = umap_coords[:, 0]
features['UMAP_2'] = umap_coords[:, 1]

# Clustering
if CLUSTER_ON == 'pca':
    pca = PCA(random_state=RANDOM_STATE)
    pca_all = pca.fit_transform(X_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = np.argmax(cumsum >= 0.99) + 1
    cluster_data = pca_all[:, :n_comp]
    print(f"Clustering on {n_comp} PCA components (99% variance)")
else:
    cluster_data = umap_coords
    print(f"Clustering on UMAP (2D)")

gmm = GaussianMixture(n_components=N_GMM_CLUSTERS, random_state=RANDOM_STATE, n_init=10,
                      covariance_type=GMM_COVARIANCE_TYPE)
features['Cluster'] = gmm.fit_predict(cluster_data)

print(f"GMM: {N_GMM_CLUSTERS} clusters (covariance: {GMM_COVARIANCE_TYPE})")
print(f"\nCluster distribution:")
for c, count in sorted(features['Cluster'].value_counts().items()):
    print(f"  Cluster {c}: {count}")

# ============================================================================
# 8.5 UMAP Association Analysis
# ============================================================================
print("\n8.5 UMAP ASSOCIATION ANALYSIS")
print("-" * 80)

association_results = []
variables_to_analyze = ([CATEGORICAL_VAR_COL] if CATEGORICAL_VAR_COL is not None else []) + (ADDITIONAL_CATEGORICAL_VARS if ADDITIONAL_CATEGORICAL_VARS else [])

# Filter to variables that exist in features
variables_to_analyze = [v for v in variables_to_analyze if v in features.columns]

if variables_to_analyze:
    for var_name in variables_to_analyze:
        metrics = calculate_umap_association_metrics(features, ['UMAP_1', 'UMAP_2'], var_name)
        if metrics is not None:
            association_results.append(metrics)
            sig_mark = "✓" if metrics['Significant'] else "✗"
            print(f"  {var_name:35s}: F={metrics['F_Statistic']:8.3f}, p={metrics['P_Value']:.2e}, R²={metrics['R_Squared']:.4f}, Sil={metrics['Silhouette_Score']:6.3f} {sig_mark}")

    # Create DataFrame and sort by R-squared (effect size)
    if association_results:
        assoc_df = pd.DataFrame(association_results).sort_values('R_Squared', ascending=False)

        # Save to CSV
        assoc_csv_path = f'{OUTPUT_DIR}/data/umap_association_summary.csv'
        assoc_df.to_csv(assoc_csv_path, index=False)
        print(f"\n✓ Saved association summary: {assoc_csv_path}")

        # Print formatted table
        print("\nUMAP Association Summary (sorted by effect size):")
        print("-" * 100)
        print(f"{'Variable':<35} {'N_Cat':>6} {'F_Stat':>10} {'P_Value':>12} {'R²':>8} {'Silhouette':>12} {'Sig':>5}")
        print("-" * 100)
        for _, row in assoc_df.iterrows():
            sig = "Yes" if row['Significant'] else "No"
            print(f"{row['Variable']:<35} {row['N_Categories']:>6.0f} {row['F_Statistic']:>10.3f} {row['P_Value']:>12.2e} {row['R_Squared']:>8.4f} {row['Silhouette_Score']:>12.3f} {sig:>5}")
        print("-" * 100)

        # Create Figure 1: R-squared (Effect Size)
        fig, ax = plt.subplots(figsize=(10, max(5, len(assoc_df) * 0.4)))
        colors_r2 = ['#1f77b4' if sig else '#d3d3d3' for sig in assoc_df['Significant']]
        ax.barh(range(len(assoc_df)), assoc_df['R_Squared'], color=colors_r2, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(assoc_df)))
        ax.set_yticklabels(assoc_df['Variable'], fontsize=10)
        ax.set_xlabel('R² (Effect Size)', fontweight='bold', fontsize=12)
        ax.set_title('UMAP Association: Effect Size (Variance Explained)', fontweight='bold', fontsize=13)
        ax.axvline(x=0.10, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Moderate threshold (0.10)')
        ax.axvline(x=0.25, color='darkred', linestyle='--', alpha=0.5, linewidth=2, label='Strong threshold (0.25)')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/umap_associations/umap_association_effect_size.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        # Create Figure 2: Silhouette Score
        fig, ax = plt.subplots(figsize=(10, max(5, len(assoc_df) * 0.4)))
        colors_sil = ['#1f77b4' if sig else '#d3d3d3' for sig in assoc_df['Significant']]
        ax.barh(range(len(assoc_df)), assoc_df['Silhouette_Score'], color=colors_sil, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(assoc_df)))
        ax.set_yticklabels(assoc_df['Variable'], fontsize=10)
        ax.set_xlabel('Silhouette Score', fontweight='bold', fontsize=12)
        ax.set_title('UMAP Association: Cluster Separation (Silhouette)', fontweight='bold', fontsize=13)
        ax.axvline(x=0.25, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Good threshold (0.25)')
        ax.axvline(x=0.50, color='darkgreen', linestyle='--', alpha=0.5, linewidth=2, label='Excellent threshold (0.50)')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/umap_associations/umap_association_silhouette.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved association plots:")
        print(f"  - umap_associations/umap_association_effect_size.png")
        print(f"  - umap_associations/umap_association_silhouette.png")
else:
    print("  (No categorical variables to analyze)")

# ============================================================================
# 9. Prepare Data for Plotting
# ============================================================================
print("\n9. PREPARING DATA FOR VISUALIZATION")
print("-" * 80)

# Generate distinct colors for clusters
colors = distinctipy.distinctipy.get_colors(N_GMM_CLUSTERS)
colors = ['#{:02x}{:02x}{:02x}'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]

# Calculate UMAP axis limits (for consistency across all UMAP plots)
umap_1_min, umap_1_max = features['UMAP_1'].min(), features['UMAP_1'].max()
umap_2_min, umap_2_max = features['UMAP_2'].min(), features['UMAP_2'].max()
umap_1_pad = (umap_1_max - umap_1_min) * 0.05
umap_2_pad = (umap_2_max - umap_2_min) * 0.05
umap_xlim = [umap_1_min - umap_1_pad, umap_1_max + umap_1_pad]
umap_ylim = [umap_2_min - umap_2_pad, umap_2_max + umap_2_pad]

print(f"UMAP axis limits: x={umap_xlim}, y={umap_ylim}")

# ============================================================================
# 9a. Cluster Selection for Trajectory Analysis (if CLUSTER_TRAJ=True)
# ============================================================================
SELECTED_CLUSTER = None  # Will be set if CLUSTER_TRAJ=True
umap_xlim_traj = None  # Will be set if CLUSTER_TRAJ=True
umap_ylim_traj = None  # Will be set if CLUSTER_TRAJ=True

if CLUSTER_TRAJ:
    print("\n9a. CLUSTER SELECTION FOR TRAJECTORY ANALYSIS")
    print("-" * 80)

    # Display cluster distribution
    print("\nAvailable clusters:")
    cluster_counts = {}
    for c in sorted(features['Cluster'].unique()):
        count = (features['Cluster'] == c).sum()
        cluster_counts[c] = count
        print(f"  Cluster {c}: {count} samples")

    # Create and display cluster UMAP plot
    print("\nDisplaying cluster UMAP plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    total_samples = len(features)
    for c in sorted(features['Cluster'].unique()):
        mask = features['Cluster'] == c
        n_in_group = mask.sum()
        pct_in_group = (n_in_group / total_samples) * 100
        cluster_label = f'Cluster {c} (n={n_in_group}, {pct_in_group:.1f}%)'
        ax.scatter(features.loc[mask, 'UMAP_1'], features.loc[mask, 'UMAP_2'],
                  label=cluster_label, s=80, alpha=0.7, color=colors[int(c)], edgecolor='black', linewidth=0.5)
    ax.set_title(f'GMM Clusters (Select one for trajectory analysis)', fontweight='bold', fontsize=14)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_xlim(umap_xlim)
    ax.set_ylim(umap_ylim)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()

    # Get user input
    valid_selection = False
    while not valid_selection:
        try:
            user_input = input(f"\nEnter cluster number for trajectory analysis (0-{N_GMM_CLUSTERS-1}): ").strip()
            SELECTED_CLUSTER = int(user_input)

            if SELECTED_CLUSTER not in cluster_counts:
                print(f"ERROR: Cluster {SELECTED_CLUSTER} not found. Valid clusters: {list(sorted(cluster_counts.keys()))}")
                continue

            n_cluster_samples = cluster_counts[SELECTED_CLUSTER]
            if n_cluster_samples < 10:
                print(f"ERROR: Cluster {SELECTED_CLUSTER} has only {n_cluster_samples} samples. Minimum 10 required for trajectory analysis.")
                continue

            valid_selection = True
            print(f"\n✓ Selected Cluster {SELECTED_CLUSTER} for trajectory analysis (n={n_cluster_samples})")

        except ValueError:
            print(f"ERROR: Please enter a valid integer between 0 and {N_GMM_CLUSTERS-1}")

    plt.close()

    # Calculate cluster-specific UMAP limits for trajectory plots only
    print(f"\nCalculating cluster-specific UMAP limits for trajectory plots...")
    cluster_mask = features['Cluster'] == SELECTED_CLUSTER
    cluster_features = features[cluster_mask]

    umap_1_min_cluster = cluster_features['UMAP_1'].min()
    umap_1_max_cluster = cluster_features['UMAP_1'].max()
    umap_2_min_cluster = cluster_features['UMAP_2'].min()
    umap_2_max_cluster = cluster_features['UMAP_2'].max()

    umap_1_pad_cluster = (umap_1_max_cluster - umap_1_min_cluster) * 0.05
    umap_2_pad_cluster = (umap_2_max_cluster - umap_2_min_cluster) * 0.05

    umap_xlim_traj = [umap_1_min_cluster - umap_1_pad_cluster, umap_1_max_cluster + umap_1_pad_cluster]
    umap_ylim_traj = [umap_2_min_cluster - umap_2_pad_cluster, umap_2_max_cluster + umap_2_pad_cluster]

    print(f"✓ Trajectory plots will use cluster-specific limits")
    print(f"  X limits: {umap_xlim_traj}")
    print(f"  Y limits: {umap_ylim_traj}")

# ============================================================================
# 10. Fit Survival Curves
# ============================================================================
if clinical is not None:
    print("\n10. FITTING SURVIVAL CURVES")
    print("-" * 80)

    # Include exploratory variables in surv dataframe
    surv_cols_to_include = ['Cluster', surv_time_col, surv_event_col]
    if CATEGORICAL_VAR_COL is not None:
        surv_cols_to_include.insert(0, CATEGORICAL_VAR_COL)
    surv_cols_to_include.extend([var for var in ADDITIONAL_CATEGORICAL_VARS if var in features.columns])
    surv = features[surv_cols_to_include].dropna()
    print(f"Samples with survival data: {len(surv)}")

    # Fit survival curves for each category
    kmf_fits_cat = []
    if CATEGORICAL_VAR_COL is not None:
        for label in sorted(surv[CATEGORICAL_VAR_COL].unique()):
            mask = surv[CATEGORICAL_VAR_COL] == label
            kmf_temp = KaplanMeierFitter()
            kmf_temp.fit(surv.loc[mask, surv_time_col], surv.loc[mask, surv_event_col], label=label)
            kmf_fits_cat.append(kmf_temp)

    # Fit survival curves for each cluster
    kmf_fits_clust = []
    for c in sorted(surv['Cluster'].unique()):
        mask = surv['Cluster'] == c
        kmf_temp = KaplanMeierFitter()
        kmf_temp.fit(surv.loc[mask, surv_time_col], surv.loc[mask, surv_event_col], label=f'C{c}')
        kmf_fits_clust.append(kmf_temp)

    # NOTE: Y-axis limits will be calculated after trajectory analysis (Section 11)
    # to include risk group curves. This will be done at the end of Section 11.

    # ============================================================================
    # 11. Trajectory Analysis - Cox Regression & Risk Groups
    # ============================================================================
    print("\n11. UMAP TRAJECTORY ANALYSIS")
    print("-" * 80)

    # Prepare data with survival, UMAP coordinates, and categorical variable
    traj_cols = ['UMAP_1', 'UMAP_2', surv_time_col, surv_event_col] + ([CATEGORICAL_VAR_COL] if CATEGORICAL_VAR_COL is not None else [])
    if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
        # Filter to selected cluster for trajectory analysis
        traj_data = features[features['Cluster'] == SELECTED_CLUSTER][traj_cols].dropna()
        print(f"Samples for trajectory analysis (Cluster {SELECTED_CLUSTER}): {len(traj_data)}")
    else:
        # Use all samples for trajectory analysis (default)
        traj_data = features[traj_cols].dropna()
        print(f"Samples for trajectory analysis: {len(traj_data)}")

    # Initialize variables for trajectory analysis results
    traj_data_copy = None
    selected_model = None
    selected_idx = None
    models_list = []
    kmf_fits_risk = []
    risk_group_sort_order = {}
    umap_scaler = None
    cmap_risk = None
    colors_risk = None
    y_lim_risk = None

    if len(traj_data) >= 10:
        # Normalize UMAP coordinates
        umap_scaler = StandardScaler()
        umap_scaled = umap_scaler.fit_transform(traj_data[['UMAP_1', 'UMAP_2']].values)

        # Create base UMAP data (scaled coordinates only)
        umap_base_df = pd.DataFrame({
            'UMAP_1': umap_scaled[:, 0],
            'UMAP_2': umap_scaled[:, 1]
        }, index=traj_data.index)

        # Fit Cox models for polynomial degrees 1 through POLYNOMIAL_DEGREE
        models_list = []
        interaction_str = "with" if INCLUDE_INTERACTION_TERMS else "without"
        print(f"\n  Fitting polynomial models up to degree {POLYNOMIAL_DEGREE} ({interaction_str} interaction terms):")

        for degree in range(1, POLYNOMIAL_DEGREE + 1):
            # Generate polynomial features for this degree
            poly_features = generate_polynomial_features(umap_base_df, degree, include_interaction=INCLUDE_INTERACTION_TERMS)

            # Create model data with survival columns + polynomial features
            model_data = pd.DataFrame({
                surv_time_col: traj_data[surv_time_col].values,
                surv_event_col: traj_data[surv_event_col].values
            }, index=traj_data.index)

            # Add all polynomial features
            for col in poly_features.columns:
                model_data[col] = poly_features[col].values

            # Fit Cox model
            cph = CoxPHFitter()
            try:
                cph.fit(model_data, duration_col=surv_time_col, event_col=surv_event_col)
                c_index = cph.concordance_index_

                # Count number of features (excluding survival columns)
                n_features = len([c for c in model_data.columns if c not in [surv_time_col, surv_event_col]])

                # Create model name with descriptive label
                interaction_label = "+interaction" if INCLUDE_INTERACTION_TERMS else ""
                if degree == 1 and INCLUDE_INTERACTION_TERMS:
                    model_name = f'Degree {degree} (linear+interaction, {n_features} features)'
                elif degree == 1:
                    model_name = f'Degree {degree} (linear, {n_features} features)'
                elif degree == 2 and INCLUDE_INTERACTION_TERMS:
                    model_name = f'Degree {degree} (quadratic+interaction, {n_features} features)'
                elif degree == 2:
                    model_name = f'Degree {degree} (quadratic, no interaction, {n_features} features)'
                elif degree == 3 and INCLUDE_INTERACTION_TERMS:
                    model_name = f'Degree {degree} (cubic+interaction, {n_features} features)'
                elif degree == 3:
                    model_name = f'Degree {degree} (cubic, no interaction, {n_features} features)'
                else:
                    model_name = f'Degree {degree} ({n_features} features{interaction_label})'

                models_list.append({
                    'Model': model_name,
                    'Degree': degree,
                    'C-index': c_index,
                    'Fitted': cph,
                    'Data': model_data,
                    'N_features': n_features
                })
                print(f"    Degree {degree}: C-index = {c_index:.4f} ({n_features} features{interaction_label})")
            except Exception as e:
                print(f"    Degree {degree} failed: {e}")

        # Select model
        if len(models_list) > 0:
            if TRAJECTORY_MODEL == 'best':
                # Select model with highest C-index
                best_idx = max(range(len(models_list)), key=lambda i: models_list[i]['C-index'])
                selected_model = models_list[best_idx]
                selected_idx = best_idx
                print(f"\n  Selection mode: 'best' → Selected degree {selected_model['Degree']}")
            elif isinstance(TRAJECTORY_MODEL, int):
                # User specified a degree - find the model with that degree
                matching_models = [i for i, m in enumerate(models_list) if m['Degree'] == TRAJECTORY_MODEL]
                if matching_models:
                    selected_idx = matching_models[0]
                    selected_model = models_list[selected_idx]
                    print(f"\n  Selection mode: degree {TRAJECTORY_MODEL} (user-specified)")
                else:
                    # Fallback to best if specified degree doesn't exist
                    print(f"\n  WARNING: Degree {TRAJECTORY_MODEL} not found (max available: {max(m['Degree'] for m in models_list)})")
                    print(f"  Falling back to 'best' selection")
                    best_idx = max(range(len(models_list)), key=lambda i: models_list[i]['C-index'])
                    selected_model = models_list[best_idx]
                    selected_idx = best_idx
            else:
                # Fallback to best
                best_idx = max(range(len(models_list)), key=lambda i: models_list[i]['C-index'])
                selected_model = models_list[best_idx]
                selected_idx = best_idx
                print(f"\n  Selection mode: default → 'best' (degree {selected_model['Degree']})")

            print(f"  Selected: {selected_model['Model']} (C-index: {selected_model['C-index']:.4f})")

            # Print coefficients for UMAP model
            umap_coef = selected_model['Fitted'].summary.copy()
            umap_coef['HR'] = np.exp(umap_coef['coef'])
            umap_coef['HR_95CI_lower'] = np.exp(umap_coef['coef'] - 1.96 * umap_coef['se(coef)'])
            umap_coef['HR_95CI_upper'] = np.exp(umap_coef['coef'] + 1.96 * umap_coef['se(coef)'])

            print(f"  Coefficients:")
            for idx, row in umap_coef.iterrows():
                sig_marker = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
                print(f"    {str(idx):25s}: HR={row['HR']:6.3f} 95%CI=[{row['HR_95CI_lower']:6.3f}-{row['HR_95CI_upper']:6.3f}] p={row['p']:.4f} {sig_marker}")

            # Generate risk scores
            risk_scores = selected_model['Fitted'].predict_partial_hazard(selected_model['Data'].drop(columns=[surv_time_col, surv_event_col]))
            traj_data_copy = traj_data.copy()
            traj_data_copy['Risk_Score'] = risk_scores.values

            # Stratify into risk groups using either threshold or quantile method
            print(f"\n  Creating risk groups (method={RISK_GROUP_METHOD}, N={N_RISK_GROUPS}):")

            if RISK_GROUP_METHOD == 'quantile':
                # Quantile-based binning: Equal-sized groups
                # Uses pd.qcut to create bins with approximately equal number of samples
                print(f"    Method: Quantile-based (equal-sized groups)")
                print(f"    HR range: {risk_scores.min():.3f} to {risk_scores.max():.3f}")

                # Create quantile-based bins
                traj_data_copy['Risk_Group'] = pd.qcut(risk_scores, q=N_RISK_GROUPS,
                                                        labels=False, duplicates='drop')

                # Get actual bin edges for labeling
                bin_edges = pd.qcut(risk_scores, q=N_RISK_GROUPS, retbins=True, duplicates='drop')[1]

                # Create descriptive labels based on actual bin ranges
                labels = []
                for i in range(len(bin_edges) - 1):
                    lower = bin_edges[i]
                    upper = bin_edges[i + 1]
                    if i == 0:
                        labels.append(f'<{upper:.2f}x (Lowest Risk)')
                    elif i == len(bin_edges) - 2:
                        labels.append(f'>{lower:.2f}x (Highest Risk)')
                    else:
                        labels.append(f'{lower:.2f}-{upper:.2f}x')

                # Replace numeric labels with descriptive labels
                traj_data_copy['Risk_Group'] = traj_data_copy['Risk_Group'].astype(int).map(dict(enumerate(labels))).astype('category')

                # Print group sizes
                print(f"    Quantile bin edges: {[f'{e:.2f}' for e in bin_edges]}")
                for group in labels:
                    count = (traj_data_copy['Risk_Group'] == group).sum()
                    pct = (count / len(traj_data_copy)) * 100
                    print(f"      {group}: {count} ({pct:.1f}%)")

            else:  # threshold-based
                # Threshold-based binning: Fixed risk score thresholds (may create imbalanced groups)
                print(f"    Method: Threshold-based (symmetric on log scale, BASE_POWER={BASE_POWER})")

                # Convert to log scale for symmetric spacing
                log_risk_scores = np.log(risk_scores)

                # Calculate the range on log scale
                log_min = np.log(risk_scores.min())
                log_max = np.log(risk_scores.max())
                max_span = max(abs(log_min), abs(log_max))

                # Create linearly spaced points on normalized scale [-1, 1]
                normalized_thresholds = np.linspace(-1, 1, N_RISK_GROUPS + 1)

                # Apply inverse power transformation to exaggerate extremes (if BASE_POWER > 1.0)
                transformed_thresholds = np.sign(normalized_thresholds) * np.abs(normalized_thresholds) ** (1 / BASE_POWER)

                # Scale back to log space (keeping full range)
                log_thresholds = transformed_thresholds * max_span

                # Convert back to linear scale
                thresholds = np.exp(log_thresholds[1:-1])

                # Print thresholds
                print(f"    HR range: {risk_scores.min():.3f} to {risk_scores.max():.3f}")
                print(f"    Thresholds: {[f'{t:.2f}' for t in thresholds]}")

                # Generate risk group labels with HR ranges
                labels = []
                for i in range(N_RISK_GROUPS):
                    if i == 0:
                        lower = risk_scores.min()
                        upper = thresholds[0]
                        labels.append(f'<{upper:.2f}x (Lowest Risk)')
                    elif i == N_RISK_GROUPS - 1:
                        lower = thresholds[-1]
                        upper = risk_scores.max()
                        labels.append(f'>{lower:.2f}x (Highest Risk)')
                    else:
                        lower = thresholds[i-1]
                        upper = thresholds[i]
                        labels.append(f'{lower:.2f}-{upper:.2f}x')

                # Stratify using adaptive symmetric thresholds
                traj_data_copy['Risk_Group'] = pd.cut(risk_scores,
                                                       bins=[-np.inf] + list(thresholds) + [np.inf],
                                                       labels=labels,
                                                       include_lowest=True)

                # Print group sizes
                for group in labels:
                    count = (traj_data_copy['Risk_Group'] == group).sum()
                    pct = (count / len(traj_data_copy)) * 100
                    print(f"      {group}: {count} ({pct:.1f}%)")

            # Create sort order map for categorical risk groups (to preserve category order, not lexicographic)
            risk_group_sort_order = {label: i for i, label in enumerate(traj_data_copy['Risk_Group'].cat.categories)}

            # Create colormap with gradient and wide spacing for risk groups
            # Use a gradient with more intermediate colors for better separation between adjacent groups
            # Colors: dark blue -> cyan -> green -> yellow -> orange -> red
            risk_colors = ['#0D47A1', '#0288D1', '#00BCD4', '#4CAF50', '#FFC107', '#FF6F00', '#D32F2F']
            cmap_risk = LinearSegmentedColormap.from_list('risk', risk_colors, N=256)
            colors_risk = cmap_risk(np.linspace(0, 1, N_RISK_GROUPS))

            # Calculate y-limits for risk groups
            kmf_fits_risk = []
            for group in sorted(traj_data_copy['Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order[x]):
                mask = traj_data_copy['Risk_Group'] == group
                kmf_temp = KaplanMeierFitter()
                kmf_temp.fit(traj_data_copy.loc[mask, surv_time_col], traj_data_copy.loc[mask, surv_event_col], label=group)
                kmf_fits_risk.append(kmf_temp)

            # Store risk groups back to features for combined plots
            features['Risk_Group'] = pd.Series(dtype='object')
            features.loc[traj_data_copy.index, 'Risk_Group'] = traj_data_copy['Risk_Group']
            # Also propagate the continuous Risk_Score so the figure-5
            # cache exporter can save the EXACT values used here, rather
            # than have downstream code re-derive them from polynomial
            # coefficients (which would re-introduce stochasticity).
            features['Risk_Score'] = pd.Series(dtype='float64')
            features.loc[traj_data_copy.index, 'Risk_Score'] = traj_data_copy['Risk_Score']

            # Save model comparison
            models_comp = pd.DataFrame([
                {'Model': m['Model'], 'C-index': m['C-index']}
                for m in models_list
            ]).sort_values('C-index', ascending=False)
            models_comp.to_csv(f'{OUTPUT_DIR}/data/trajectory_models.csv', index=False)

    else:
        print("WARNING: Not enough samples for trajectory analysis")

    # ============================================================================
    # Calculate GLOBAL Y-Limits for ALL Survival Plots
    # ============================================================================
    print("\nCalculating global y-axis limits for survival plots...")

    # Collect all survival values from all curve types
    all_y_values = []
    all_y_values.extend([kmf.survival_function_.values.flatten() for kmf in kmf_fits_cat])
    all_y_values.extend([kmf.survival_function_.values.flatten() for kmf in kmf_fits_clust])

    # Include risk group curves if they were created
    if kmf_fits_risk:
        all_y_values.extend([kmf.survival_function_.values.flatten() for kmf in kmf_fits_risk])

    all_y_values = np.concatenate(all_y_values)
    y_min_global, y_max_global = all_y_values.min(), all_y_values.max()

    # Add symmetric 5% padding below and above
    y_range = y_max_global - y_min_global
    y_pad_bottom = y_range * 0.05
    y_pad_top = y_range * 0.05

    # Calculate limits with padding (allow limits to exceed [0,1] for visual spacing)
    y_lim_surv = [y_min_global - y_pad_bottom, y_max_global + y_pad_top]

    print(f"Global survival y-limits (all panels): {y_lim_surv}")

    # ============================================================================
    # 12. PLOTTING
    # ============================================================================
    print("\n12. PLOTTING")
    print("-" * 80)

    # --- Individual UMAP Plots ---
    print("\nCreating individual UMAP plots...")

    # UMAP: Categorical
    if CATEGORICAL_VAR_COL is not None:
        print("  - UMAP: Categorical")
        fig, ax = plt.subplots(figsize=(8, 7))
        total_samples = len(features)
        for label in sorted(features[CATEGORICAL_VAR_COL].unique()):
            mask = features[CATEGORICAL_VAR_COL] == label
            n_in_group = mask.sum()
            pct_in_group = (n_in_group / total_samples) * 100
            label_with_count = f"{label} (n={n_in_group}, {pct_in_group:.1f}%)"
            ax.scatter(features.loc[mask, 'UMAP_1'], features.loc[mask, 'UMAP_2'],
                      label=label_with_count, s=50, alpha=0.6)
        ax.set_title(f'UMAP: {CATEGORICAL_VAR_COL}', fontweight='bold', fontsize=12)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_xlim(umap_xlim)
        ax.set_ylim(umap_ylim)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/umap_individual/umap_categorical.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

    # UMAP: Clusters
    print("  - UMAP: Clusters")
    fig, ax = plt.subplots(figsize=(8, 7))
    total_samples = len(features)
    for c in sorted(features['Cluster'].unique()):
        mask = features['Cluster'] == c
        n_in_group = mask.sum()
        pct_in_group = (n_in_group / total_samples) * 100
        cluster_label = f'Cluster {c} (n={n_in_group}, {pct_in_group:.1f}%)'
        ax.scatter(features.loc[mask, 'UMAP_1'], features.loc[mask, 'UMAP_2'],
                  label=cluster_label, s=50, alpha=0.6, color=colors[int(c)])
    ax.set_title(f'UMAP: GMM Clusters ({N_GMM_CLUSTERS})', fontweight='bold', fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_xlim(umap_xlim)
    ax.set_ylim(umap_ylim)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/umap_individual/umap_clusters.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # UMAP: Survival Time (all patients colored by time; censored optional)
    if surv_time_col is not None and surv_event_col is not None:
        event_mask = features[surv_event_col] == 1
        censor_mask = (~event_mask) & features[surv_time_col].notna()
        plot_mask = features[surv_time_col].notna() if SURV_TIME_UMAP_SHOW_CENSORED else event_mask
        n_events = event_mask.sum()
        n_censored = censor_mask.sum()
        print(f"  - UMAP: Survival Time ({'events + censored' if SURV_TIME_UMAP_SHOW_CENSORED else 'events only'})")

        surv_cmap = cmap_risk.reversed() if cmap_risk is not None else 'plasma_r'
        # Use a shared color scale across events and censored so the colorbar is meaningful
        vmin = features.loc[plot_mask, surv_time_col].min()
        vmax = features.loc[plot_mask, surv_time_col].max()

        fig, ax = plt.subplots(figsize=(8, 7))
        if SURV_TIME_UMAP_SHOW_CENSORED and censor_mask.sum() > 0:
            ax.scatter(features.loc[censor_mask, 'UMAP_1'], features.loc[censor_mask, 'UMAP_2'],
                      c=features.loc[censor_mask, surv_time_col], cmap=surv_cmap,
                      vmin=vmin, vmax=vmax, s=30, alpha=0.4, marker='^',
                      edgecolor='none')
        if event_mask.sum() > 0:
            sc = ax.scatter(features.loc[event_mask, 'UMAP_1'], features.loc[event_mask, 'UMAP_2'],
                           c=features.loc[event_mask, surv_time_col], cmap=surv_cmap,
                           vmin=vmin, vmax=vmax, s=50, alpha=0.85,
                           edgecolor='white', linewidth=0.4)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Survival Time (days)', fontsize=10)
        title_suffix = f'events n={n_events}' + (f', censored n={n_censored}' if SURV_TIME_UMAP_SHOW_CENSORED else '')
        ax.set_title(f'UMAP: Survival Time ({title_suffix})', fontweight='bold', fontsize=12)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_xlim(umap_xlim)
        ax.set_ylim(umap_ylim)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/umap_individual/umap_survival_time.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

    # UMAP: Risk Groups (if trajectory analysis was performed)
    if traj_data_copy is not None and selected_model is not None:
        print("  - UMAP: Risk Groups")
        fig, ax = plt.subplots(figsize=(8, 7))
        total_risk_samples = len(traj_data_copy)
        for group in sorted(traj_data_copy['Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order[x]):
            group_idx = risk_group_sort_order[group]  # Use sort order to get color index
            mask = traj_data_copy['Risk_Group'] == group
            n_in_group = mask.sum()
            pct_in_group = (n_in_group / total_risk_samples) * 100
            group_label = f"{group} (n={n_in_group}, {pct_in_group:.1f}%)"
            ax.scatter(features.loc[traj_data_copy[mask].index, 'UMAP_1'],
                      features.loc[traj_data_copy[mask].index, 'UMAP_2'],
                      label=group_label, s=50, alpha=0.8, edgecolor='white', linewidth=1.5, color=colors_risk[group_idx])
        if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
            title = f'UMAP: Risk Groups (Cluster {SELECTED_CLUSTER})'
            xlim_to_use = umap_xlim_traj
            ylim_to_use = umap_ylim_traj
        else:
            title = 'UMAP: Risk Groups'
            xlim_to_use = umap_xlim
            ylim_to_use = umap_ylim
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_xlim(xlim_to_use)
        ax.set_ylim(ylim_to_use)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/umap_individual/umap_risk_groups.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        # UMAP: Risk Score (continuous gradient)
        print("  - UMAP: Risk Score Gradient")
        fig, ax = plt.subplots(figsize=(8, 7))
        scatter = ax.scatter(features.loc[traj_data_copy.index, 'UMAP_1'],
                            features.loc[traj_data_copy.index, 'UMAP_2'],
                            c=traj_data_copy['Risk_Score'], cmap=cmap_risk, s=50, alpha=0.7,
                            edgecolor='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, label='Risk Score (Proportional Hazard Ratio)')
        n_samples = len(traj_data_copy)
        if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
            title = f'UMAP: Risk Score Gradient (Cluster {SELECTED_CLUSTER}, n={n_samples})'
        else:
            title = f'UMAP: Risk Score Gradient (n={n_samples})'
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
            xlim_to_use = umap_xlim_traj
            ylim_to_use = umap_ylim_traj
        else:
            xlim_to_use = umap_xlim
            ylim_to_use = umap_ylim
        ax.set_xlim(xlim_to_use)
        ax.set_ylim(ylim_to_use)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/umap_individual/umap_risk_score_gradient.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

    # --- Combined UMAP Plot ---
    print("\nCreating combined UMAP plot...")
    umap_panels_to_plot = COMBINED_UMAP_PANELS.copy()
    # Only include risk_groups if trajectory analysis was performed
    if traj_data_copy is None and 'risk_groups' in umap_panels_to_plot:
        umap_panels_to_plot.remove('risk_groups')
    # Only include categorical panel if CATEGORICAL_VAR_COL is set
    if CATEGORICAL_VAR_COL is None and 'categorical' in umap_panels_to_plot:
        umap_panels_to_plot.remove('categorical')

    n_umap_panels = len(umap_panels_to_plot)
    if n_umap_panels > 0:
        fig_width = 6 + (n_umap_panels - 1) * 6
        fig, axes = plt.subplots(1, n_umap_panels, figsize=(fig_width, 6))
        if n_umap_panels == 1:
            axes = [axes]

        for ax_idx, panel in enumerate(umap_panels_to_plot):
            ax = axes[ax_idx]

            if panel == 'categorical':
                total_samples = len(features)
                for label in sorted(features[CATEGORICAL_VAR_COL].unique()):
                    mask = features[CATEGORICAL_VAR_COL] == label
                    n_in_group = mask.sum()
                    pct_in_group = (n_in_group / total_samples) * 100
                    label_with_count = f"{label} (n={n_in_group}, {pct_in_group:.1f}%)"
                    ax.scatter(features.loc[mask, 'UMAP_1'], features.loc[mask, 'UMAP_2'],
                              label=label_with_count, s=40, alpha=0.6)
                ax.set_title(f'UMAP: {CATEGORICAL_VAR_COL}', fontweight='bold')

            elif panel == 'clusters':
                total_samples = len(features)
                for c in sorted(features['Cluster'].unique()):
                    mask = features['Cluster'] == c
                    n_in_group = mask.sum()
                    pct_in_group = (n_in_group / total_samples) * 100
                    cluster_label = f'Cluster {c} (n={n_in_group}, {pct_in_group:.1f}%)'
                    ax.scatter(features.loc[mask, 'UMAP_1'], features.loc[mask, 'UMAP_2'],
                              label=cluster_label, s=40, alpha=0.6, color=colors[int(c)])
                ax.set_title(f'UMAP: GMM Clusters ({N_GMM_CLUSTERS})', fontweight='bold')

            elif panel == 'risk_groups' and traj_data_copy is not None:
                total_risk_samples = len(traj_data_copy)
                for group in sorted(traj_data_copy['Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order[x]):
                    group_idx = risk_group_sort_order[group]
                    mask = traj_data_copy['Risk_Group'] == group
                    n_in_group = mask.sum()
                    pct_in_group = (n_in_group / total_risk_samples) * 100
                    group_label = f"{group} (n={n_in_group}, {pct_in_group:.1f}%)"
                    ax.scatter(features.loc[traj_data_copy[mask].index, 'UMAP_1'],
                              features.loc[traj_data_copy[mask].index, 'UMAP_2'],
                              label=group_label, s=50, alpha=0.8, edgecolor='white', linewidth=1.5, color=colors_risk[group_idx])
                if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
                    title = f'UMAP: Risk Groups (Cluster {SELECTED_CLUSTER})'
                else:
                    title = 'UMAP: Risk Groups'
                ax.set_title(title, fontweight='bold')

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            if panel == 'risk_groups' and CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
                xlim_to_use = umap_xlim_traj
                ylim_to_use = umap_ylim_traj
            else:
                xlim_to_use = umap_xlim
                ylim_to_use = umap_ylim
            ax.set_xlim(xlim_to_use)
            ax.set_ylim(ylim_to_use)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.2)
            ax.set_aspect('equal', adjustable='datalim')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/umap_combined.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved: umap_combined.png ({n_umap_panels} panels)")

    # --- Cluster-Categorical Variable Concordance Analysis ---
    if CATEGORICAL_VAR_COL is not None:
        print("\nCreating concordance analysis (Clusters vs Categorical Variable)...")

        # Create contingency table
        contingency_table = pd.crosstab(features['Cluster'], features[CATEGORICAL_VAR_COL])

        # ---- Absolute Counts Heatmap ----
        print("  - Concordance table: Absolute counts")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Sample Count'},
                    ax=ax, linewidths=0.5, linecolor='gray')
        ax.set_title(f'Cluster-{CATEGORICAL_VAR_COL} Concordance: Absolute Counts', fontweight='bold', fontsize=12)
        ax.set_xlabel(CATEGORICAL_VAR_COL, fontweight='bold')
        ax.set_ylabel('Cluster', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/concordance_counts.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        # ---- Percentage Heatmap ----
        print("  - Concordance table: Percentages (row-normalized)")
        # Normalize by rows (cluster) to show percentage within each cluster
        contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(contingency_pct, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage (%)'},
                    ax=ax, linewidths=0.5, linecolor='gray')
        ax.set_title(f'Cluster-{CATEGORICAL_VAR_COL} Concordance: Row Percentages (%)', fontweight='bold', fontsize=12)
        ax.set_xlabel(CATEGORICAL_VAR_COL, fontweight='bold')
        ax.set_ylabel('Cluster', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/concordance_percentages.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        # Print summary statistics
        print(f"\nConcordance Summary:")
        print(f"  Total samples: {contingency_table.sum().sum()}")
        print(f"  Clusters: {len(contingency_table)}")
        print(f"  {CATEGORICAL_VAR_COL} categories: {len(contingency_table.columns)}")
        print(f"  Saved: concordance_counts.png and concordance_percentages.png")

    # --- Initialize Model Comparison Tracking ---
    # Dictionary to store metrics for all models for final comparison CSV
    model_comparison_metrics = {}

    # --- Individual Survival Plots ---
    print("\nCreating individual survival plots...")

    # Survival: Categorical
    if CATEGORICAL_VAR_COL is not None:
        print("  - Survival: Categorical")
        fig, ax = plt.subplots(figsize=(8, 6))
        total_samples = len(surv)
        # Build label mapping with sample counts
        cat_labels_with_counts = []
        for label in sorted(surv[CATEGORICAL_VAR_COL].unique()):
            mask = surv[CATEGORICAL_VAR_COL] == label
            n_in_group = mask.sum()
            pct_in_group = (n_in_group / total_samples) * 100
            label_with_count = f"{label} (n={n_in_group}, {pct_in_group:.1f}%)"
            cat_labels_with_counts.append(label_with_count)
        # Plot with updated labels
        for kmf, label_with_count in zip(kmf_fits_cat, cat_labels_with_counts):
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, label=label_with_count)

        # Fit Cox model for categorical variable to get risk scores
        try:
            cox_cat = CoxPHFitter()
            cat_cox_data = surv[[CATEGORICAL_VAR_COL, surv_time_col, surv_event_col]].copy()
            dummy_cat = pd.get_dummies(cat_cox_data[CATEGORICAL_VAR_COL], drop_first=True, prefix='cat')
            cox_cat_data = cat_cox_data[[surv_time_col, surv_event_col]].copy()
            for col in dummy_cat.columns:
                cox_cat_data[col] = dummy_cat[col].values
            cox_cat.fit(cox_cat_data, duration_col=surv_time_col, event_col=surv_event_col)
            cat_risk_scores = pd.Series(cox_cat.predict_partial_hazard(cox_cat_data), index=cox_cat_data.index)
        except Exception as e:
            print(f"    Warning: Could not fit Cox model for {CATEGORICAL_VAR_COL}: {e}")
            cat_risk_scores = None
            cox_cat = None

        # Compute comprehensive metrics
        print(f"\n  === SUBTYPE SURVIVAL ANALYSIS DEBUG ===")
        print(f"  Cohort size for Subtype analysis: {len(surv)}")
        print(f"  Patients in cohort: {list(surv.index[:10])}..." if len(surv) > 10 else f"  Patients: {list(surv.index)}")
        print(f"  Subtype groups: {sorted(surv[CATEGORICAL_VAR_COL].unique())}")
        print(f"  Group counts: {surv[CATEGORICAL_VAR_COL].value_counts().to_dict()}")
        if cat_risk_scores is not None:
            print(f"  Cat Risk Scores - min: {cat_risk_scores.min():.4f}, max: {cat_risk_scores.max():.4f}, mean: {cat_risk_scores.mean():.4f}, std: {cat_risk_scores.std():.4f}")
        print(f"  === END SUBTYPE DEBUG ===\n")

        metrics_cat = compute_comprehensive_survival_metrics(
            surv, CATEGORICAL_VAR_COL, surv_time_col, surv_event_col, SURVIVAL_EVAL_TIMEPOINTS,
            cox_model=cox_cat,
            risk_scores=cat_risk_scores
        )
        # Create title with log-rank p-value
        logrank_p_str = f"{metrics_cat['logrank_p']:.4f}" if metrics_cat['logrank_p'] < 0.0001 else f"{metrics_cat['logrank_p']:.4f}"
        title = f'{SURV_EVENT_COL} Survival: {CATEGORICAL_VAR_COL}\n'
        title += f'C-index: {metrics_cat["c_index"]:.3f}, Log-rank p: {logrank_p_str}'
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(y_lim_surv)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/survival_individual/survival_categorical.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        # Save metrics to CSV
        print(f"    C-index: {metrics_cat['c_index']:.4f}, Log-rank p: {logrank_p_str}")
        metrics_cat_df = pd.DataFrame({
            'Variable': CATEGORICAL_VAR_COL,
            'C_index': metrics_cat['c_index'],
            'Log_rank_p': metrics_cat['logrank_p'],
            'Log_rank_stat': metrics_cat['logrank_stat']
        }, index=[0])
        # Add median survival and survival at timepoints
        for group, median_surv in metrics_cat['median_survival'].items():
            metrics_cat_df[f'Median_survival_{group}'] = median_surv
        for t, survs_at_t in metrics_cat['survival_at_t'].items():
            for group, surv_prob in survs_at_t.items():
                metrics_cat_df[f'Survival_{t}d_{group}'] = surv_prob
        metrics_cat_df.to_csv(f'{OUTPUT_DIR}/data/survival_metrics/metrics_{CATEGORICAL_VAR_COL}.csv', index=False)

        # Add to model comparison tracking
        model_comparison_metrics[CATEGORICAL_VAR_COL] = {
            'C_index': metrics_cat['c_index'],
            'Log_rank_p': metrics_cat['logrank_p'],
            'Log_rank_stat': metrics_cat['logrank_stat'],
            'Median_survival': metrics_cat['median_survival'],
            'Survival_at_timepoints': metrics_cat['survival_at_t'],
            'AUC_at_timepoints': metrics_cat.get('auc_at_t', {}),
            'AUC_integrated': metrics_cat.get('auc_integrated', np.nan),
            'N_groups': len(metrics_cat['median_survival']),
            'N_samples': len(surv)
        }

    # Survival: Clusters
    print("  - Survival: Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    total_samples = len(surv)
    # Build label mapping with sample counts
    clust_labels_with_counts = []
    for c in sorted(surv['Cluster'].unique()):
        mask = surv['Cluster'] == c
        n_in_group = mask.sum()
        pct_in_group = (n_in_group / total_samples) * 100
        label_with_count = f'Cluster {c} (n={n_in_group}, {pct_in_group:.1f}%)'
        clust_labels_with_counts.append(label_with_count)
    # Plot with updated labels
    for c, (kmf, label_with_count) in enumerate(zip(kmf_fits_clust, clust_labels_with_counts)):
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color=colors[c], label=label_with_count)

    # Fit Cox model for clusters to get risk scores
    try:
        cox_clust = CoxPHFitter()
        clust_cox_data = surv[['Cluster', surv_time_col, surv_event_col]].copy()
        dummy_clust = pd.get_dummies(clust_cox_data['Cluster'], drop_first=True, prefix='clust')
        cox_clust_data = clust_cox_data[[surv_time_col, surv_event_col]].copy()
        for col in dummy_clust.columns:
            cox_clust_data[col] = dummy_clust[col].values
        cox_clust.fit(cox_clust_data, duration_col=surv_time_col, event_col=surv_event_col)
        clust_risk_scores = pd.Series(cox_clust.predict_partial_hazard(cox_clust_data), index=cox_clust_data.index)
    except Exception as e:
        print(f"    Warning: Could not fit Cox model for Cluster: {e}")
        clust_risk_scores = None
        cox_clust = None

    # Compute comprehensive metrics
    metrics_clust = compute_comprehensive_survival_metrics(
        surv, 'Cluster', surv_time_col, surv_event_col, SURVIVAL_EVAL_TIMEPOINTS,
        cox_model=cox_clust,
        risk_scores=clust_risk_scores
    )
    # Create title with log-rank p-value
    logrank_p_str = f"{metrics_clust['logrank_p']:.4f}" if metrics_clust['logrank_p'] < 0.0001 else f"{metrics_clust['logrank_p']:.4f}"
    title = f'{SURV_EVENT_COL} Survival: GMM Clusters\n'
    title += f'C-index: {metrics_clust["c_index"]:.3f}, Log-rank p: {logrank_p_str}'
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival Probability')
    ax.set_ylim(y_lim_surv)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/survival_individual/survival_clusters.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    # Save metrics to CSV
    print(f"    C-index: {metrics_clust['c_index']:.4f}, Log-rank p: {logrank_p_str}")
    metrics_clust_df = pd.DataFrame({
        'Variable': 'Cluster',
        'C_index': metrics_clust['c_index'],
        'Log_rank_p': metrics_clust['logrank_p'],
        'Log_rank_stat': metrics_clust['logrank_stat']
    }, index=[0])
    # Add median survival and survival at timepoints
    for group, median_surv in metrics_clust['median_survival'].items():
        metrics_clust_df[f'Median_survival_Cluster_{group}'] = median_surv
    for t, survs_at_t in metrics_clust['survival_at_t'].items():
        for group, surv_prob in survs_at_t.items():
            metrics_clust_df[f'Survival_{t}d_Cluster_{group}'] = surv_prob
    metrics_clust_df.to_csv(f'{OUTPUT_DIR}/data/survival_metrics/metrics_Cluster.csv', index=False)

    # Add to model comparison tracking
    model_comparison_metrics['Cluster'] = {
        'C_index': metrics_clust['c_index'],
        'Log_rank_p': metrics_clust['logrank_p'],
        'Log_rank_stat': metrics_clust['logrank_stat'],
        'Median_survival': metrics_clust['median_survival'],
        'Survival_at_timepoints': metrics_clust['survival_at_t'],
        'AUC_at_timepoints': metrics_clust.get('auc_at_t', {}),
        'AUC_integrated': metrics_clust.get('auc_integrated', np.nan),
        'N_groups': len(metrics_clust['median_survival']),
        'N_samples': len(surv)
    }

    # Survival: Risk Groups (if trajectory analysis was performed)
    if traj_data_copy is not None and selected_model is not None:
        print("  - Survival: Risk Groups")
        fig, ax = plt.subplots(figsize=(8, 6))
        total_risk_samples = len(traj_data_copy)
        for group in sorted(traj_data_copy['Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order[x]):
            group_idx = risk_group_sort_order[group]
            mask = traj_data_copy['Risk_Group'] == group
            n_in_group = mask.sum()
            pct_in_group = (n_in_group / total_risk_samples) * 100
            group_label = f"{group} (n={n_in_group}, {pct_in_group:.1f}%)"
            kmf = KaplanMeierFitter()
            kmf.fit(traj_data_copy.loc[mask, surv_time_col], traj_data_copy.loc[mask, surv_event_col], label=group_label)
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color=colors_risk[group_idx])
        # Compute comprehensive metrics for risk groups
        # Pass Cox model and risk scores so we can compute time-dependent AUC and Brier scores
        print(f"\n  === RISK GROUPS SURVIVAL ANALYSIS DEBUG ===")
        print(f"  Cohort size for Risk Groups analysis: {len(traj_data_copy)}")
        print(f"  Patients in cohort: {list(traj_data_copy.index[:10])}..." if len(traj_data_copy) > 10 else f"  Patients: {list(traj_data_copy.index)}")
        print(f"  Risk groups: {sorted(traj_data_copy['Risk_Group'].unique())}")
        print(f"  Group counts: {traj_data_copy['Risk_Group'].value_counts().to_dict()}")
        print(f"  Risk Scores from UMAP model - min: {traj_data_copy['Risk_Score'].min():.4f}, max: {traj_data_copy['Risk_Score'].max():.4f}, mean: {traj_data_copy['Risk_Score'].mean():.4f}, std: {traj_data_copy['Risk_Score'].std():.4f}")
        print(f"  C-index from selected UMAP model: {selected_model['C-index']:.4f}")
        print(f"  === END RISK GROUPS DEBUG ===\n")

        risk_scores_series = pd.Series(traj_data_copy['Risk_Score'].values, index=traj_data_copy.index)
        metrics_risk = compute_comprehensive_survival_metrics(
            traj_data_copy, 'Risk_Group', surv_time_col, surv_event_col, SURVIVAL_EVAL_TIMEPOINTS,
            cox_model=selected_model['Fitted'],
            risk_scores=risk_scores_series
        )
        # Store BOTH C-indices for Risk Groups:
        # 1. Categorical C-index (from Risk_Group discretization) - for fair comparison with Subtype
        # 2. Continuous C-index (from underlying UMAP Cox model) - for reference/transparency
        cindex_risk_categorical = metrics_risk['c_index']
        cindex_risk_continuous = selected_model['C-index']

        logrank_p_str = f"{metrics_risk['logrank_p']:.4f}" if metrics_risk['logrank_p'] < 0.0001 else f"{metrics_risk['logrank_p']:.4f}"
        if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
            title = f'{SURV_EVENT_COL} Survival: Risk Groups (Cluster {SELECTED_CLUSTER})\n'
            title += f'C-index: {cindex_risk_categorical:.3f} (cat), {cindex_risk_continuous:.3f} (cont), Log-rank p: {logrank_p_str}'
        else:
            title = f'{SURV_EVENT_COL} Survival: Risk Groups\n'
            title += f'C-index: {cindex_risk_categorical:.3f} (cat), {cindex_risk_continuous:.3f} (cont), Log-rank p: {logrank_p_str}'
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(y_lim_surv)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/survival_individual/survival_risk_groups.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        # Save metrics to CSV
        print(f"    C-index (categorical): {cindex_risk_categorical:.4f}, C-index (continuous): {cindex_risk_continuous:.4f}, Log-rank p: {logrank_p_str}")
        metrics_risk_df = pd.DataFrame({
            'Variable': 'Risk_Group',
            'C_index_categorical': cindex_risk_categorical,
            'C_index_continuous': cindex_risk_continuous,
            'Log_rank_p': metrics_risk['logrank_p'],
            'Log_rank_stat': metrics_risk['logrank_stat']
        }, index=[0])
        # Add median survival and survival at timepoints
        for group, median_surv in metrics_risk['median_survival'].items():
            metrics_risk_df[f'Median_survival_{group}'] = median_surv
        for t, survs_at_t in metrics_risk['survival_at_t'].items():
            for group, surv_prob in survs_at_t.items():
                metrics_risk_df[f'Survival_{t}d_{group}'] = surv_prob
        metrics_risk_df.to_csv(f'{OUTPUT_DIR}/data/survival_metrics/metrics_Risk_Group.csv', index=False)

        # Add to model comparison tracking (with BOTH C-indices)
        model_comparison_metrics['Risk_Group'] = {
            'C_index': cindex_risk_categorical,  # Primary: categorical for fair comparison
            'C_index_continuous': cindex_risk_continuous,  # Secondary: continuous for reference
            'Log_rank_p': metrics_risk['logrank_p'],
            'Log_rank_stat': metrics_risk['logrank_stat'],
            'Median_survival': metrics_risk['median_survival'],
            'Survival_at_timepoints': metrics_risk['survival_at_t'],
            'AUC_at_timepoints': metrics_risk.get('auc_at_t', {}),
            'AUC_integrated': metrics_risk.get('auc_integrated', np.nan),
            'N_groups': len(metrics_risk['median_survival']),
            'N_samples': len(traj_data_copy)
        }

    # --- Exploratory Additional Categorical Variables ---
    if ADDITIONAL_CATEGORICAL_VARS:
        print("\nCreating exploratory plots for additional variables...")

        for var_name in ADDITIONAL_CATEGORICAL_VARS:
            # Validate variable exists
            if var_name not in features.columns:
                print(f"  WARNING: '{var_name}' not found in features data, skipping")
                continue

            # Check if variable has actual data (not all NaN/null)
            if features[var_name].isna().sum() == len(features):
                print(f"  WARNING: '{var_name}' has no data, skipping")
                continue

            print(f"  - Creating plots for '{var_name}'")

            # ---- UMAP Plot ----
            unique_vals = sorted(features[var_name].dropna().unique())
            if len(unique_vals) > 1:  # Only plot if > 1 category
                fig, ax = plt.subplots(figsize=(8, 7))
                total_samples = len(features)

                for label in unique_vals:
                    mask = features[var_name] == label
                    n_in_group = mask.sum()
                    pct_in_group = (n_in_group / total_samples) * 100
                    label_with_count = f"{label} (n={n_in_group}, {pct_in_group:.1f}%)"
                    ax.scatter(features.loc[mask, 'UMAP_1'], features.loc[mask, 'UMAP_2'],
                              label=label_with_count, s=50, alpha=0.6)

                ax.set_title(f'UMAP: {var_name}', fontweight='bold', fontsize=12)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_xlim(umap_xlim)
                ax.set_ylim(umap_ylim)
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.2)
                ax.set_aspect('equal', adjustable='datalim')
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_DIR}/plots/umap_individual/umap_exploratory_{var_name}.png', dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

            # ---- Survival Plot ----
            # Only create survival plot if variable exists in survival data and has multiple categories
            if var_name in surv.columns:
                surv_unique_vals = sorted(surv[var_name].dropna().unique())
                if len(surv_unique_vals) > 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    total_samples = len(surv)

                    for label in surv_unique_vals:
                        mask = surv[var_name] == label
                        n_in_group = mask.sum()
                        pct_in_group = (n_in_group / total_samples) * 100
                        label_with_count = f"{label} (n={n_in_group}, {pct_in_group:.1f}%)"

                        kmf = KaplanMeierFitter()
                        kmf.fit(surv.loc[mask, surv_time_col], surv.loc[mask, surv_event_col], label=label_with_count)
                        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2)

                    # Fit Cox model for exploratory variable to get risk scores
                    try:
                        cox_explor = CoxPHFitter()
                        explor_cox_data = surv[[var_name, surv_time_col, surv_event_col]].dropna().copy()
                        dummy_explor = pd.get_dummies(explor_cox_data[var_name], drop_first=True, prefix='var')
                        cox_explor_data = explor_cox_data[[surv_time_col, surv_event_col]].copy()
                        for col in dummy_explor.columns:
                            cox_explor_data[col] = dummy_explor[col].values
                        cox_explor.fit(cox_explor_data, duration_col=surv_time_col, event_col=surv_event_col)
                        explor_risk_scores = pd.Series(cox_explor.predict_partial_hazard(cox_explor_data), index=cox_explor_data.index)
                    except Exception as e:
                        print(f"    Warning: Could not fit Cox model for {var_name}: {e}")
                        explor_risk_scores = None
                        cox_explor = None

                    # Compute comprehensive metrics
                    metrics_explor = compute_comprehensive_survival_metrics(
                        surv, var_name, surv_time_col, surv_event_col, SURVIVAL_EVAL_TIMEPOINTS,
                        cox_model=cox_explor,
                        risk_scores=explor_risk_scores
                    )
                    logrank_p_str = f"{metrics_explor['logrank_p']:.4f}" if metrics_explor['logrank_p'] < 0.0001 else f"{metrics_explor['logrank_p']:.4f}"
                    title = f'{SURV_EVENT_COL} Survival: {var_name}\n'
                    title += f'C-index: {metrics_explor["c_index"]:.3f}, Log-rank p: {logrank_p_str}'
                    ax.set_title(title, fontweight='bold', fontsize=12)
                    ax.set_xlabel('Time (days)')
                    ax.set_ylabel('Survival Probability')
                    ax.set_ylim(y_lim_surv)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.2)
                    plt.tight_layout()
                    plt.savefig(f'{OUTPUT_DIR}/plots/survival_individual/survival_exploratory_{var_name}.png', dpi=PLOT_DPI, bbox_inches='tight')
                    plt.close()

                    # Add to model comparison tracking
                    model_comparison_metrics[var_name] = {
                        'C_index': metrics_explor['c_index'],
                        'Log_rank_p': metrics_explor['logrank_p'],
                        'Log_rank_stat': metrics_explor['logrank_stat'],
                        'Median_survival': metrics_explor['median_survival'],
                        'Survival_at_timepoints': metrics_explor['survival_at_t'],
                        'AUC_at_timepoints': metrics_explor.get('auc_at_t', {}),
                        'AUC_integrated': metrics_explor.get('auc_integrated', np.nan),
                        'N_groups': len(metrics_explor['median_survival']),
                        'N_samples': len(surv[surv[var_name].notna()])
                    }

            # ---- Concordance Plots ----
            # Compare exploratory variable to GMM clusters (with GMM Cluster on y-axis)
            if 'Cluster' in features.columns:
                try:
                    # Counts version
                    plot_concordance(
                        'GMM Cluster', features.loc[features[var_name].notna(), 'Cluster'],
                        var_name, features[var_name].dropna(),
                        f'{OUTPUT_DIR}/plots/concordance_plots/gmm_clusters/counts/concordance_{var_name}_vs_gmm_clusters.png',
                        percentage=False
                    )
                    # Percentages version
                    plot_concordance(
                        'GMM Cluster', features.loc[features[var_name].notna(), 'Cluster'],
                        var_name, features[var_name].dropna(),
                        f'{OUTPUT_DIR}/plots/concordance_plots/gmm_clusters/percentages/concordance_{var_name}_vs_gmm_clusters.png',
                        percentage=True
                    )
                except Exception as e:
                    print(f"    WARNING: Could not create concordance plot for {var_name} vs GMM clusters: {e}")

            # Compare exploratory variable to risk groups (with Risk Group on y-axis)
            if 'Risk_Group' in features.columns and features['Risk_Group'].notna().any():
                try:
                    # Counts version
                    plot_concordance(
                        'Risk Group', features.loc[features[var_name].notna(), 'Risk_Group'],
                        var_name, features[var_name].dropna(),
                        f'{OUTPUT_DIR}/plots/concordance_plots/risk_groups/counts/concordance_{var_name}_vs_risk_groups.png',
                        percentage=False
                    )
                    # Percentages version
                    plot_concordance(
                        'Risk Group', features.loc[features[var_name].notna(), 'Risk_Group'],
                        var_name, features[var_name].dropna(),
                        f'{OUTPUT_DIR}/plots/concordance_plots/risk_groups/percentages/concordance_{var_name}_vs_risk_groups.png',
                        percentage=True
                    )
                except Exception as e:
                    print(f"    WARNING: Could not create concordance plot for {var_name} vs risk groups: {e}")

    # --- Combined Survival Plot ---
    print("\nCreating combined survival plot...")
    survival_panels_to_plot = COMBINED_SURVIVAL_PANELS.copy()
    # Only include risk_groups if trajectory analysis was performed
    if traj_data_copy is None and 'risk_groups' in survival_panels_to_plot:
        survival_panels_to_plot.remove('risk_groups')
    # Only include categorical panel if CATEGORICAL_VAR_COL is set
    if CATEGORICAL_VAR_COL is None and 'categorical' in survival_panels_to_plot:
        survival_panels_to_plot.remove('categorical')

    n_survival_panels = len(survival_panels_to_plot)
    if n_survival_panels > 0:
        fig_width = 6 + (n_survival_panels - 1) * 6
        fig, axes = plt.subplots(1, n_survival_panels, figsize=(fig_width, 5))
        if n_survival_panels == 1:
            axes = [axes]

        for ax_idx, panel in enumerate(survival_panels_to_plot):
            ax = axes[ax_idx]
            kmf_local = KaplanMeierFitter()

            if panel == 'categorical':
                total_samples = len(surv)
                for label in sorted(surv[CATEGORICAL_VAR_COL].unique()):
                    mask = surv[CATEGORICAL_VAR_COL] == label
                    n_in_group = mask.sum()
                    pct_in_group = (n_in_group / total_samples) * 100
                    label_with_count = f"{label} (n={n_in_group}, {pct_in_group:.1f}%)"
                    kmf_local.fit(surv.loc[mask, surv_time_col], surv.loc[mask, surv_event_col], label=label_with_count)
                    kmf_local.plot_survival_function(ax=ax, ci_show=False, linewidth=2)
                # Compute comprehensive metrics
                if 'metrics_cat' not in locals() or metrics_cat is None:
                    metrics_cat = compute_comprehensive_survival_metrics(surv, CATEGORICAL_VAR_COL, surv_time_col, surv_event_col, SURVIVAL_EVAL_TIMEPOINTS)
                cindex_cat_combined = metrics_cat['c_index']
                logrank_p_cat = metrics_cat['logrank_p']
                logrank_p_str = f"{logrank_p_cat:.4f}" if logrank_p_cat < 0.0001 else f"{logrank_p_cat:.4f}"
                ax.set_title(f'{CATEGORICAL_VAR_COL}: {SURV_EVENT_COL} Survival\nC-index: {cindex_cat_combined:.3f}, Log-rank p: {logrank_p_str}', fontweight='bold')
                ax.set_ylim(y_lim_surv)

            elif panel == 'clusters':
                total_samples = len(surv)
                for c in sorted(surv['Cluster'].unique()):
                    mask = surv['Cluster'] == c
                    n_in_group = mask.sum()
                    pct_in_group = (n_in_group / total_samples) * 100
                    cluster_label = f'Cluster {c} (n={n_in_group}, {pct_in_group:.1f}%)'
                    kmf_local.fit(surv.loc[mask, surv_time_col], surv.loc[mask, surv_event_col], label=cluster_label)
                    kmf_local.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color=colors[int(c)])
                # Compute comprehensive metrics
                if 'metrics_clust' not in locals() or metrics_clust is None:
                    metrics_clust = compute_comprehensive_survival_metrics(surv, 'Cluster', surv_time_col, surv_event_col, SURVIVAL_EVAL_TIMEPOINTS)
                cindex_clust_combined = metrics_clust['c_index']
                logrank_p_clust = metrics_clust['logrank_p']
                logrank_p_str = f"{logrank_p_clust:.4f}" if logrank_p_clust < 0.0001 else f"{logrank_p_clust:.4f}"
                ax.set_title(f'GMM Clusters: {SURV_EVENT_COL} Survival\nC-index: {cindex_clust_combined:.3f}, Log-rank p: {logrank_p_str}', fontweight='bold')
                ax.set_ylim(y_lim_surv)

            elif panel == 'risk_groups' and traj_data_copy is not None:
                total_risk_samples = len(traj_data_copy)
                for group in sorted(traj_data_copy['Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order[x]):
                    group_idx = risk_group_sort_order[group]
                    mask = traj_data_copy['Risk_Group'] == group
                    n_in_group = mask.sum()
                    pct_in_group = (n_in_group / total_risk_samples) * 100
                    group_label = f"{group} (n={n_in_group}, {pct_in_group:.1f}%)"
                    kmf_local.fit(traj_data_copy.loc[mask, surv_time_col], traj_data_copy.loc[mask, surv_event_col], label=group_label)
                    kmf_local.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color=colors_risk[group_idx])
                # Report BOTH C-indices for transparency
                if 'metrics_risk' not in locals() or metrics_risk is None:
                    risk_scores_series = pd.Series(traj_data_copy['Risk_Score'].values, index=traj_data_copy.index)
                    metrics_risk = compute_comprehensive_survival_metrics(
                        traj_data_copy, 'Risk_Group', surv_time_col, surv_event_col, SURVIVAL_EVAL_TIMEPOINTS,
                        cox_model=selected_model['Fitted'],
                        risk_scores=risk_scores_series
                    )
                # Use same variable names as individual plot for consistency
                if 'cindex_risk_categorical' not in locals():
                    cindex_risk_categorical = metrics_risk['c_index']
                    cindex_risk_continuous = selected_model['C-index']

                logrank_p_risk = metrics_risk['logrank_p']
                logrank_p_str = f"{logrank_p_risk:.4f}" if logrank_p_risk < 0.0001 else f"{logrank_p_risk:.4f}"
                if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
                    title = f'Risk Groups (Cluster {SELECTED_CLUSTER}): {SURV_EVENT_COL} Survival\n'
                    title += f'C-index: {cindex_risk_categorical:.3f} (cat), {cindex_risk_continuous:.3f} (cont), Log-rank p: {logrank_p_str}'
                else:
                    title = f'Risk Groups: {SURV_EVENT_COL} Survival\n'
                    title += f'C-index: {cindex_risk_categorical:.3f} (cat), {cindex_risk_continuous:.3f} (cont), Log-rank p: {logrank_p_str}'
                ax.set_title(title, fontweight='bold')
                ax.set_ylim(y_lim_surv)

            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Survival Probability')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/survival_combined.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved: survival_combined.png ({n_survival_panels} panels)")

    # --- Trajectory Plots (if trajectory analysis was performed) ---
    if traj_data_copy is not None and selected_model is not None:
        print("\nCreating trajectory plots...")

        # Risk contour plot
        print("  - Risk contour plot")
        fig, ax = plt.subplots(figsize=(10, 8))

        umap1_grid = np.linspace(features['UMAP_1'].min(), features['UMAP_1'].max(), CONTOUR_RESOLUTION)
        umap2_grid = np.linspace(features['UMAP_2'].min(), features['UMAP_2'].max(), CONTOUR_RESOLUTION)
        X_grid, Y_grid = np.meshgrid(umap1_grid, umap2_grid)

        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        grid_scaled = umap_scaler.transform(grid_points)

        # Create grid data using same polynomial degree as selected model
        grid_base_df = pd.DataFrame({
            'UMAP_1': grid_scaled[:, 0],
            'UMAP_2': grid_scaled[:, 1]
        })

        # Generate polynomial features for grid using selected model's degree
        selected_degree = selected_model['Degree']
        grid_poly_features = generate_polynomial_features(grid_base_df, selected_degree, include_interaction=INCLUDE_INTERACTION_TERMS)

        # Predict risk on grid
        risk_grid = selected_model['Fitted'].predict_partial_hazard(grid_poly_features).values
        Z = risk_grid.reshape(X_grid.shape)

        # Plot contour
        contour = ax.contourf(X_grid, Y_grid, Z, levels=15, cmap=cmap_risk, alpha=0.7)
        cbar = plt.colorbar(contour, ax=ax, label='Risk Score')

        # Overlay samples
        total_risk_samples = len(traj_data_copy)
        for group in sorted(traj_data_copy['Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order[x]):
            group_idx = risk_group_sort_order[group]
            mask = traj_data_copy['Risk_Group'] == group
            n_in_group = mask.sum()
            pct_in_group = (n_in_group / total_risk_samples) * 100
            group_label = f"{group} (n={n_in_group}, {pct_in_group:.1f}%)"
            ax.scatter(features.loc[traj_data_copy[mask].index, 'UMAP_1'],
                      features.loc[traj_data_copy[mask].index, 'UMAP_2'],
                      label=group_label, s=50, alpha=0.8, edgecolor='white', linewidth=1.5, color=colors_risk[group_idx])

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
            title = f'Risk Trajectory - Cluster {SELECTED_CLUSTER} ({selected_model["Model"]})'
        else:
            title = f'Risk Trajectory ({selected_model["Model"]})'
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9, title='Risk Group')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
            ax.set_xlim(umap_xlim_traj)
            ax.set_ylim(umap_ylim_traj)
        else:
            ax.set_xlim(umap_xlim)
            ax.set_ylim(umap_ylim)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/risk_trajectory.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        # Risk-stratified survival curves
        print("  - Risk-stratified survival curves")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curves
        total_risk_samples = len(traj_data_copy)
        for group in sorted(traj_data_copy['Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order[x]):
            group_idx = risk_group_sort_order[group]
            mask = traj_data_copy['Risk_Group'] == group
            n_in_group = mask.sum()
            pct_in_group = (n_in_group / total_risk_samples) * 100
            group_label = f"{group} (n={n_in_group}, {pct_in_group:.1f}%)"
            kmf = KaplanMeierFitter()
            kmf.fit(traj_data_copy.loc[mask, surv_time_col], traj_data_copy.loc[mask, surv_event_col], label=group_label)
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color=colors_risk[group_idx])

        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(y_lim_surv)
        if CLUSTER_TRAJ and SELECTED_CLUSTER is not None:
            title = f'{SURV_EVENT_COL} Survival by Risk Group - Cluster {SELECTED_CLUSTER} ({selected_model["Model"]})'
        else:
            title = f'{SURV_EVENT_COL} Survival by Risk Group ({selected_model["Model"]})'
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/risk_survival.png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        # --- Risk Group - Categorical Variable Concordance Analysis ---
        print("\n  - Risk Group - Categorical Variable Concordance")

        # Create contingency table between risk groups and categorical variable
        concordance_data = traj_data_copy.copy()
        if CATEGORICAL_VAR_COL in concordance_data.columns:
            risk_cat_table = pd.crosstab(concordance_data['Risk_Group'],
                                          concordance_data[CATEGORICAL_VAR_COL])

            # ---- Absolute Counts Heatmap ----
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(risk_cat_table, annot=True, fmt='d', cmap='Blues',
                       cbar_kws={'label': 'Sample Count'},
                       ax=ax, linewidths=0.5, linecolor='gray')
            ax.set_title(f'Risk Groups - {CATEGORICAL_VAR_COL} Concordance: Absolute Counts',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel(CATEGORICAL_VAR_COL, fontweight='bold')
            ax.set_ylabel('Risk Group', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/plots/risk_concordance_counts.png', dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()

            # ---- Percentage Heatmap ----
            risk_cat_pct = risk_cat_table.div(risk_cat_table.sum(axis=1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(risk_cat_pct, annot=True, fmt='.1f', cmap='Blues',
                       cbar_kws={'label': 'Percentage (%)'},
                       ax=ax, linewidths=0.5, linecolor='gray')
            ax.set_title(f'Risk Groups - {CATEGORICAL_VAR_COL} Concordance: Row Percentages (%)',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel(CATEGORICAL_VAR_COL, fontweight='bold')
            ax.set_ylabel('Risk Group', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/plots/risk_concordance_percentages.png', dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Saved: risk_concordance_counts.png and risk_concordance_percentages.png")

    print("\nAll plots saved successfully!")

    # ============================================================================
    # 13. Cox Comparison
    # ============================================================================
    print("\n13. COX MODEL COMPARISON")
    print("-" * 80)
    cph = CoxPHFitter()
    results = {}

    compare_cols = ([CATEGORICAL_VAR_COL] if CATEGORICAL_VAR_COL is not None else []) + ['Cluster', surv_time_col, surv_event_col]
    surv_compare = features[compare_cols].dropna()
    print(f"\nSurvival data shape: {surv_compare.shape}")

    # Categorical variable
    if CATEGORICAL_VAR_COL is not None:
        try:
            cox_cat = surv_compare[[surv_time_col, surv_event_col, CATEGORICAL_VAR_COL]].copy()
            cat_dum = pd.get_dummies(cox_cat[CATEGORICAL_VAR_COL], drop_first=True)
            cat_dum_all = pd.get_dummies(cox_cat[CATEGORICAL_VAR_COL], drop_first=False)
            reference_cat = [col for col in cat_dum_all.columns if col not in cat_dum.columns][0]
            cox_cat_fit = pd.concat([cox_cat[[surv_time_col, surv_event_col]], cat_dum], axis=1)

            cph.fit(cox_cat_fit, duration_col=surv_time_col, event_col=surv_event_col)
            results[CATEGORICAL_VAR_COL] = cph.concordance_index_
            print(f"\n{CATEGORICAL_VAR_COL}: C-index = {results[CATEGORICAL_VAR_COL]:.4f}")

            # Print coefficients for categorical model
            cat_coef = cph.summary.copy()
            cat_coef['HR'] = np.exp(cat_coef['coef'])
            cat_coef['HR_95CI_lower'] = np.exp(cat_coef['coef'] - 1.96 * cat_coef['se(coef)'])
            cat_coef['HR_95CI_upper'] = np.exp(cat_coef['coef'] + 1.96 * cat_coef['se(coef)'])

            # Add reference category
            ref_row_cat = pd.Series({
                'coef': 0.0, 'exp(coef)': 1.0, 'se(coef)': 0.0,
                'coef lower 95%': 0.0, 'coef upper 95%': 0.0,
                'exp(coef) lower 95%': 1.0, 'exp(coef) upper 95%': 1.0,
                'z': 0.0, 'p': 1.0, '-log2(p)': 0.0,
                'HR': 1.0, 'HR_95CI_lower': 1.0, 'HR_95CI_upper': 1.0
            }, name=reference_cat)
            cat_coef = pd.concat([pd.DataFrame([ref_row_cat]), cat_coef])

            print(f"  Coefficients:")
            for idx, row in cat_coef.iterrows():
                sig_marker = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
                ref_marker = " [REFERENCE]" if idx == reference_cat else ""
                print(f"    {str(idx):25s}: HR={row['HR']:6.3f} 95%CI=[{row['HR_95CI_lower']:6.3f}-{row['HR_95CI_upper']:6.3f}] p={row['p']:.4f} {sig_marker}{ref_marker}")
        except Exception as e:
            print(f"\n{CATEGORICAL_VAR_COL} Cox failed: {e}")

    # Clusters
    try:
        cox_clust = surv_compare[[surv_time_col, surv_event_col, 'Cluster']].copy()
        clust_dum = pd.get_dummies(cox_clust['Cluster'], drop_first=True)
        cox_clust_fit = pd.concat([cox_clust[[surv_time_col, surv_event_col]], clust_dum], axis=1)

        cph.fit(cox_clust_fit, duration_col=surv_time_col, event_col=surv_event_col)
        results['GMM Clusters'] = cph.concordance_index_
        print(f"GMM Clusters: C-index = {results['GMM Clusters']:.4f}")
    except Exception as e:
        print(f"GMM Clusters Cox failed: {e}")

    # Summary
    print("\nC-Index Comparison (higher = better):")
    comp = pd.DataFrame(list(results.items()), columns=['Model', 'C-index']).sort_values('C-index', ascending=False)
    for _, row in comp.iterrows():
        print(f"  {row['Model']:20s}: {row['C-index']:.4f}")

    comp.to_csv(f'{OUTPUT_DIR}/data/cox_comparison.csv', index=False)

    # ============================================================================
    # 14. Combined Cox Model: Genomic (UMAP) + Categorical
    # ============================================================================
    if CATEGORICAL_VAR_COL is not None:
        print("\n" + "="*80)
        print("14. COMBINED COX MODEL: GENOMIC (UMAP) + CATEGORICAL")
        print("="*80)
        print("\nTesting if UMAP coordinates and categorical variable are independent prognosticators")

        try:
            # Prepare data for combined model
            combined_data = features[['UMAP_1', 'UMAP_2', CATEGORICAL_VAR_COL, surv_time_col, surv_event_col]].dropna()
            print(f"\nCombined model data: {combined_data.shape[0]} samples with complete data")

            # Normalize UMAP coordinates for combined model (same as Section 11)
            umap_scaler_combined = StandardScaler()
            umap_scaled_combined = umap_scaler_combined.fit_transform(combined_data[['UMAP_1', 'UMAP_2']].values)

            # Create base data with normalized UMAP and categorical variable
            combined_model_base = pd.DataFrame({
                surv_time_col: combined_data[surv_time_col].values,
                surv_event_col: combined_data[surv_event_col].values,
                'UMAP_1': umap_scaled_combined[:, 0],
                'UMAP_2': umap_scaled_combined[:, 1]
            }, index=combined_data.index)

            # One-hot encode categorical variable (drop_first=True to avoid multicollinearity, but we'll add reference back for display)
            cat_dum = pd.get_dummies(combined_data[CATEGORICAL_VAR_COL], drop_first=True)
            cat_dum_all = pd.get_dummies(combined_data[CATEGORICAL_VAR_COL], drop_first=False)
            reference_cat = [col for col in cat_dum_all.columns if col not in cat_dum.columns][0]

            for col in cat_dum.columns:
                combined_model_base[col] = cat_dum[col].values

            # Use same polynomial degree as selected trajectory model
            print(f"\n  Fitting combined model using selected trajectory model (degree {selected_model['Degree']}):")

            # Generate polynomial features for UMAP coordinates
            umap_only_df = combined_model_base[['UMAP_1', 'UMAP_2']].copy()
            poly_features_combined = generate_polynomial_features(umap_only_df, selected_model['Degree'],
                                                                  include_interaction=INCLUDE_INTERACTION_TERMS)

            # Create combined model data: survival cols + polynomial UMAP features + categorical dummies
            cox_combined_fit = pd.DataFrame({
                surv_time_col: combined_model_base[surv_time_col].values,
                surv_event_col: combined_model_base[surv_event_col].values
            }, index=combined_model_base.index)

            # Add polynomial UMAP features
            for col in poly_features_combined.columns:
                cox_combined_fit[col] = poly_features_combined[col].values

            # Add categorical dummy variables
            for col in cat_dum.columns:
                cox_combined_fit[col] = cat_dum[col].values

            # Create model name
            n_umap_features = len(poly_features_combined.columns)
            n_cat_features = len(cat_dum.columns)
            if selected_model['Degree'] == 1:
                model_name = f'Combined: Polynomial deg {selected_model["Degree"]} (linear) + {CATEGORICAL_VAR_COL} ({n_umap_features}+{n_cat_features} features)'
            elif selected_model['Degree'] == 2:
                model_name = f'Combined: Polynomial deg {selected_model["Degree"]} (quadratic) + {CATEGORICAL_VAR_COL} ({n_umap_features}+{n_cat_features} features)'
            elif selected_model['Degree'] == 3:
                model_name = f'Combined: Polynomial deg {selected_model["Degree"]} (cubic) + {CATEGORICAL_VAR_COL} ({n_umap_features}+{n_cat_features} features)'
            else:
                model_name = f'Combined: Polynomial deg {selected_model["Degree"]} + {CATEGORICAL_VAR_COL} ({n_umap_features}+{n_cat_features} features)'

            # Fit the selected combined model
            cph_combined = CoxPHFitter()
            cph_combined.fit(cox_combined_fit, duration_col=surv_time_col, event_col=surv_event_col)
            combined_cindex = cph_combined.concordance_index_

            print(f"  {model_name}: C-index = {combined_cindex:.4f}")

            # Print coefficient table for combined model (including reference category)
            print("\n--- COMBINED MODEL COEFFICIENTS ---")
            print("(Testing independence of UMAP and categorical variables)")
            coef_summary = cph_combined.summary.copy()
            coef_summary['HR'] = np.exp(coef_summary['coef'])
            coef_summary['HR_95CI_lower'] = np.exp(coef_summary['coef'] - 1.96 * coef_summary['se(coef)'])
            coef_summary['HR_95CI_upper'] = np.exp(coef_summary['coef'] + 1.96 * coef_summary['se(coef)'])

            # Add reference category with coefficient 0 (coef=0, HR=1.0, p=1.0 - reference)
            ref_row = pd.Series({
                'coef': 0.0,
                'exp(coef)': 1.0,
                'se(coef)': 0.0,
                'coef lower 95%': 0.0,
                'coef upper 95%': 0.0,
                'exp(coef) lower 95%': 1.0,
                'exp(coef) upper 95%': 1.0,
                'z': 0.0,
                'p': 1.0,
                '-log2(p)': 0.0,
                'HR': 1.0,
                'HR_95CI_lower': 1.0,
                'HR_95CI_upper': 1.0
            }, name=reference_cat)
            coef_summary = pd.concat([pd.DataFrame([ref_row]), coef_summary])

            for idx, row in coef_summary.iterrows():
                sig_marker = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
                ref_marker = " [REFERENCE]" if idx == reference_cat else ""
                print(f"  {str(idx):30s}: coef={row['coef']:7.4f}  HR={row['HR']:6.3f}  "
                      f"95%CI=[{row['HR_95CI_lower']:6.3f}-{row['HR_95CI_upper']:6.3f}]  "
                      f"p={row['p']:.4f} {sig_marker}{ref_marker}")

            # Save combined model coefficients
            coef_summary.to_csv(f'{OUTPUT_DIR}/data/combined_model_coefficients.csv')

            # Compare C-indices
            print(f"\n--- C-INDEX COMPARISON ---")
            # Use actual C-index from selected trajectory model (Section 11) if available
            if selected_model is not None:
                umap_model_name = f'UMAP ({selected_model["Model"]})'
                results[umap_model_name] = selected_model['C-index']
            else:
                results['UMAP (trajectory model)'] = np.nan
            results['Combined (UMAP + Category)'] = combined_cindex

            comp_combined = pd.DataFrame(list(results.items()), columns=['Model', 'C-index']).sort_values('C-index', ascending=False)
            for _, row in comp_combined.iterrows():
                print(f"  {row['Model']:30s}: {row['C-index']:.4f}")

            # Interpretation
            print("\n--- INTERPRETATION ---")
            umap_sig = any(coef_summary.loc[['UMAP_1', 'UMAP_2'], 'p'] < 0.05)
            # Check all category columns (including reference) for significance
            cat_cols_all = [col for col in coef_summary.index if col not in ['UMAP_1', 'UMAP_2']]
            cat_sig = any(coef_summary.loc[cat_cols_all, 'p'] < 0.05) if len(cat_cols_all) > 0 else False

            if umap_sig and cat_sig:
                print("✓ Both UMAP and categorical variables are significant (p<0.05)")
                print("  → Independent prognosticators: genomic features ADD value beyond categorical classification")
            elif umap_sig:
                print("✓ Only UMAP is significant (p<0.05)")
                print("  → Categorical variable not significant after accounting for genomic features")
            elif cat_sig:
                print("✓ Only categorical is significant (p<0.05)")
                print("  → UMAP coordinates not significant after accounting for categorical variable")
            else:
                print("⚠ Neither UMAP nor categorical variables are individually significant")

            # Generate combined risk scores
            print("\n--- GENERATING COMBINED RISK STRATA ---")
            combined_risk_scores = cph_combined.predict_partial_hazard(cox_combined_fit.drop(columns=[surv_time_col, surv_event_col]))
            print(f"  Method: {RISK_GROUP_METHOD}")
            print(f"  HR range: {combined_risk_scores.min():.3f} to {combined_risk_scores.max():.3f}")

            # Create risk groups using same method as individual trajectory analysis
            if RISK_GROUP_METHOD == 'quantile':
                # Quantile-based binning for combined model
                combined_data['Combined_Risk_Group'] = pd.qcut(combined_risk_scores, q=N_RISK_GROUPS,
                                                                labels=False, duplicates='drop')

                # Get actual bin edges for labeling
                bin_edges_combined = pd.qcut(combined_risk_scores, q=N_RISK_GROUPS, retbins=True, duplicates='drop')[1]

                # Create descriptive labels
                labels_combined = []
                for i in range(len(bin_edges_combined) - 1):
                    lower = bin_edges_combined[i]
                    upper = bin_edges_combined[i + 1]
                    if i == 0:
                        labels_combined.append(f'<{upper:.2f}x (Lowest Risk)')
                    elif i == len(bin_edges_combined) - 2:
                        labels_combined.append(f'>{lower:.2f}x (Highest Risk)')
                    else:
                        labels_combined.append(f'{lower:.2f}-{upper:.2f}x')

                # Replace numeric labels with descriptive labels
                combined_data['Combined_Risk_Group'] = combined_data['Combined_Risk_Group'].astype(int).map(dict(enumerate(labels_combined))).astype('category')

                print(f"  Quantile bin edges: {[f'{e:.2f}' for e in bin_edges_combined]}")

            else:  # threshold-based
                # Threshold-based binning for combined model
                log_risk_combined = np.log(combined_risk_scores)
                log_min_combined = np.log(combined_risk_scores.min())
                log_max_combined = np.log(combined_risk_scores.max())

                log_center_combined = (log_min_combined + log_max_combined) / 2
                max_span_combined = max(abs(log_min_combined - log_center_combined), abs(log_max_combined - log_center_combined))

                normalized_thresholds_combined = np.linspace(-1, 1, N_RISK_GROUPS + 1)
                transformed_thresholds_combined = np.sign(normalized_thresholds_combined) * np.abs(normalized_thresholds_combined) ** (1 / BASE_POWER)
                log_thresholds_combined = log_center_combined + transformed_thresholds_combined * max_span_combined

                log_thresholds_combined = np.clip(log_thresholds_combined, log_min_combined, log_max_combined)
                thresh_combined = np.exp(log_thresholds_combined[1:-1])

                print(f"  Thresholds: {[f'{t:.2f}' for t in thresh_combined]}")

                # Generate risk group labels
                labels_combined = []
                for i in range(N_RISK_GROUPS):
                    if i == 0:
                        lower = combined_risk_scores.min()
                        upper = thresh_combined[0]
                        labels_combined.append(f'<{upper:.2f}x (Lowest Risk)')
                    elif i == N_RISK_GROUPS - 1:
                        lower = thresh_combined[-1]
                        upper = combined_risk_scores.max()
                        labels_combined.append(f'>{lower:.2f}x (Highest Risk)')
                    else:
                        lower = thresh_combined[i-1]
                        upper = thresh_combined[i]
                        labels_combined.append(f'{lower:.2f}-{upper:.2f}x')

                # Assign risk groups
                combined_data['Combined_Risk_Group'] = pd.cut(combined_risk_scores,
                                                               bins=[-np.inf] + list(thresh_combined) + [np.inf],
                                                               labels=labels_combined,
                                                               include_lowest=True)

            # Create sort order map (to preserve category order)
            risk_group_sort_order_combined = {label: i for i, label in enumerate(combined_data['Combined_Risk_Group'].cat.categories)}

            print(f"\n  Combined risk group distribution:")
            for group in sorted(combined_data['Combined_Risk_Group'].dropna().unique(), key=lambda x: risk_group_sort_order_combined[x]):
                count = (combined_data['Combined_Risk_Group'] == group).sum()
                pct = (count / len(combined_data)) * 100
                print(f"    {group:30s}: {count:4d} ({pct:5.1f}%)")

            # Create survival curves by combined risk groups
            print("\n--- CREATING COMBINED RISK GROUP SURVIVAL CURVES ---")
            fig, ax = plt.subplots(figsize=(10, 6))

            for group in sorted(combined_data['Combined_Risk_Group'].dropna().unique(),
                               key=lambda x: risk_group_sort_order_combined[x]):
                group_idx = risk_group_sort_order_combined[group]
                mask = combined_data['Combined_Risk_Group'] == group
                n_in_group = mask.sum()
                pct_in_group = (n_in_group / len(combined_data)) * 100
                group_label = f"{group} (n={n_in_group}, {pct_in_group:.1f}%)"

                kmf = KaplanMeierFitter()
                kmf.fit(combined_data.loc[mask, surv_time_col],
                       combined_data.loc[mask, surv_event_col],
                       label=group_label)
                kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2.5,
                                           color=colors_risk[group_idx] if 'colors_risk' in locals() else None)

            # Use C-index from the underlying Cox model fit to UMAP coordinates + categorical variable
            # (Risk groups are just a discretization of the continuous model, so we report the true model C-index)
            ax.set_title(f'{SURV_EVENT_COL} Survival by Combined Risk Groups\n(UMAP Coordinates + {CATEGORICAL_VAR_COL}) (C-index: {combined_cindex:.3f})',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Survival Probability')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/plots/combined_model_survival.png', dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: combined_model_survival.png")

            # Save combined model results
            comp_combined.to_csv(f'{OUTPUT_DIR}/data/combined_model_cindices.csv', index=False)

            print(f"\n✓ Combined model analysis complete. Results saved to {OUTPUT_DIR}/")

        except Exception as e:
            print(f"\n⚠ Combined model fitting failed: {e}")
            import traceback
            traceback.print_exc()

    # --- Create Consolidated Model Comparison CSV ---
    print("\nCreating consolidated model comparison CSV...")
    if model_comparison_metrics:
        # Build comparison dataframe with one row per model
        comparison_rows = []

        for model_name, metrics in model_comparison_metrics.items():
            row = {
                'Model': model_name,
                'C_index': metrics['C_index'],
                'Log_rank_p': metrics['Log_rank_p'],
                'Log_rank_stat': metrics['Log_rank_stat'],
                'N_groups': metrics['N_groups'],
                'N_samples': metrics['N_samples'],
            }

            # Add continuous C-index if available (for Risk Groups)
            if 'C_index_continuous' in metrics:
                row['C_index_continuous'] = metrics['C_index_continuous']

            # Add time-dependent AUC(t) at key timepoints
            if metrics['AUC_at_timepoints']:
                for t in SURVIVAL_EVAL_TIMEPOINTS:
                    if t in metrics['AUC_at_timepoints']:
                        auc_val = metrics['AUC_at_timepoints'][t]
                        row[f'AUC_{t}d'] = auc_val if not np.isnan(auc_val) else np.nan

            # Add integrated AUC(t) across all timepoints
            if 'AUC_integrated' in metrics:
                row['AUC_integrated'] = metrics['AUC_integrated']

            comparison_rows.append(row)

        # Create dataframe and save
        comparison_df = pd.DataFrame(comparison_rows)

        # Fill any missing AUC columns with NaN (for models that don't have them)
        for t in SURVIVAL_EVAL_TIMEPOINTS:
            if f'AUC_{t}d' not in comparison_df.columns:
                comparison_df[f'AUC_{t}d'] = np.nan

        # Fill integrated AUC column if missing
        if 'AUC_integrated' not in comparison_df.columns:
            comparison_df['AUC_integrated'] = np.nan

        comparison_df = comparison_df.sort_values('C_index', ascending=False).reset_index(drop=True)

        comparison_csv_path = f'{OUTPUT_DIR}/data/survival_metrics/model_comparison.csv'
        comparison_df.to_csv(comparison_csv_path, index=False)

        print(f"\nColumns in CSV: {list(comparison_df.columns)}")

        print(f"✓ Model comparison CSV saved: {comparison_csv_path}")
        print(f"\nModel Performance Summary (sorted by C-index):")
        print("=" * 120)
        # Only show columns that exist (include continuous C-index for Risk Groups)
        cols_to_show = ['Model', 'C_index', 'C_index_continuous', 'Log_rank_p', 'N_groups', 'N_samples']
        cols_to_show = [c for c in cols_to_show if c in comparison_df.columns]
        print(comparison_df[cols_to_show].to_string(index=False))
        print("=" * 120)

    print(f"\nResults saved to {OUTPUT_DIR}/")

else:
    print("\n10-14. SURVIVAL ANALYSIS (SKIPPED)")
    print("-" * 80)
    print("(Clinical data not provided - skipping survival analysis, trajectory, and Cox models)")
    print("\n✓ Clustering and UMAP visualization complete!")
    print(f"  Samples: {len(features)}")
    print(f"  Clusters: {len(features['Cluster'].unique())}")

# ============================================================================
# EXPORT SAMPLE ASSIGNMENTS
# ============================================================================
print("\n" + "="*80)
print("EXPORTING SAMPLE ASSIGNMENTS")
print("="*80)

# Export all sample assignments for downstream analysis
sample_assignments = pd.DataFrame(index=features.index)
sample_assignments['Patient_ID'] = features['Patient_ID']
sample_assignments['Cluster'] = features['Cluster']

# Add Risk_Group if it exists
if 'Risk_Group' in features.columns:
    sample_assignments['Risk_Group'] = features['Risk_Group']

# Export to CSV
assignments_path = f'{OUTPUT_DIR}/data/sample_assignments.csv'
sample_assignments.to_csv(assignments_path, index=False)
print(f"✓ Exported {len(sample_assignments)} sample assignments to: {assignments_path}")
print(f"  Columns: {list(sample_assignments.columns)}")
if 'Risk_Group' in sample_assignments.columns:
    print(f"  Risk groups: {sorted(sample_assignments['Risk_Group'].dropna().unique())}")
    for rg in sorted(sample_assignments['Risk_Group'].dropna().unique()):
        count = (sample_assignments['Risk_Group'] == rg).sum()
        pct = (count / len(sample_assignments)) * 100
        print(f"    {rg}: {count} ({pct:.1f}%)")
print(f"  Clusters: {sorted(sample_assignments['Cluster'].unique())}")
for cl in sorted(sample_assignments['Cluster'].unique()):
    count = (sample_assignments['Cluster'] == cl).sum()
    pct = (count / len(sample_assignments)) * 100
    print(f"    {cl}: {count} ({pct:.1f}%)")


# ============================================================================
# CACHE EXPORT FOR FIGURE 5 RENDERING
# ============================================================================
# Per-sample DataFrame with the columns needed to render the Figure 5
# panels: UMAP coords, GMM cluster, polynomial-Cox risk score / risk
# group, the categorical subtype label, and survival.
# One CSV per cohort, written alongside the existing sample_assignments.csv.
print("\n" + "=" * 80)
print("EXPORTING FIGURE 5 CACHE")
print("=" * 80)

_fig5_cols = [PATIENT_ID_COL, 'UMAP_1', 'UMAP_2', CATEGORICAL_VAR_COL,
              'Cluster']
# Risk_Score (continuous) and Risk_Group (categorical) are saved together
# so figure5 reads frozen values from the run that produced this cache,
# never re-derives them downstream.
if 'Risk_Score' in features.columns:
    _fig5_cols.append('Risk_Score')
if 'Risk_Group' in features.columns:
    _fig5_cols.append('Risk_Group')
# Survival columns (may have been renamed/derived during preprocessing)
for _c in [surv_time_col, surv_event_col]:
    if _c in features.columns and _c not in _fig5_cols:
        _fig5_cols.append(_c)

_fig5_cache = features[_fig5_cols].copy()
_fig5_cache_path = f'{OUTPUT_DIR}/data/fig5_cache.csv'
_fig5_cache.to_csv(_fig5_cache_path, index=False)
print(f"✓ Wrote per-sample cache to: {_fig5_cache_path}")
print(f"  Rows: {len(_fig5_cache)}")
print(f"  Columns: {list(_fig5_cache.columns)}")
