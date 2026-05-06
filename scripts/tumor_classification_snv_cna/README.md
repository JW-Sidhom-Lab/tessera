# SNV+CNA multimodal tumor-type classification (Figure 3)

Trains an ensemble of MLP tumor-type classifiers on top of concatenated
TESSERA SNV + CNA sample-level features, evaluated under macro-fold
nested cross-validation. Reports the multimodal-classifier panel in
Figure 3 against the SNV-only and CNA-only baselines.

## Pipeline

```
data/TCGA_PanCan/clinical.csv                    (patient -> tumour type)
scripts/data/tcga/{train,valid}_data_snv.csv     (SNV sample IDs + tumour types)
scripts/tcga_pancan_snv/var_features/            (TESSERA per-variant features)
scripts/tcga_pancan_cna/cna_features/            (TESSERA per-segment features)
                       │
                       ▼
              tumor_type_classifier_snv_cna.py    (one SNV+CNA combination)
                       │
                       ▼
              models_macro/snv_cna_ensemble_results_<SNV>_<CNA>.pkl
                       │
                       ▼
              plot_from_results.py
                       │
                       ▼
              plots/
```

## Manuscript configuration

The published multimodal classifier uses a single feature combination:

- **SNV**: `global_25` features from
  [`scripts/tcga_pancan_snv/`](../tcga_pancan_snv/README.md)
- **CNA**: `attn_2` features from
  [`scripts/tcga_pancan_cna/`](../tcga_pancan_cna/README.md)

Default training mode is **macro-fold nested CV** (5 outer / 10 inner
folds) so each sample is tested exactly once.

## Running

```bash
# Manuscript default: global_25 + attn_2
./run_snv_cna_classifier.sh

# Train one specific combination
SNV_MODEL_NAME=local_25 CNA_MODEL_NAME=attn_0 \
    python tumor_type_classifier_snv_cna.py

# Per-combination figures
SNV_MODEL_NAME=global_25 CNA_MODEL_NAME=attn_2 \
    python plot_from_results.py
```

The classifier reads:

- per-variant TESSERA SNV features from
  [`scripts/tcga_pancan_snv/var_features/`](../tcga_pancan_snv/README.md)
  (produced by `get_variant_features.py`)
- per-segment TESSERA CNA features from
  [`scripts/tcga_pancan_cna/cna_features/`](../tcga_pancan_cna/README.md)
  (produced by `get_cna_features.py`)
- SNV sample/tumour-type metadata from
  [`scripts/data/tcga/{train,valid}_data_snv.csv`](../data/tcga/README.md)
- patient-level tumour types from
  [`data/TCGA_PanCan/clinical.csv`](../../data/TCGA_PanCan/README.md)
  (used to label the CNA-only samples that have no SNV row)

so the upstream SNV + CNA pretraining and feature extraction must have
run first.

### Recognised env vars

| Variable | Notes |
|---|---|
| `SNV_MODEL_NAME` | Variant model whose features to use (default `global_25`) |
| `CNA_MODEL_NAME` | CNA attention model whose features to use (default `attn_2`) |
| `N_FOLDS`, `N_MACRO_FOLDS`, `N_INNER_FOLDS` | CV depth (10 / 5 / 10 by default) |
| `USE_MACRO_NESTED_CV`, `USE_FULL_DATASET_CV` | CV mode toggles |
| `TEST_SIZE` | Hold-out fraction when both CV modes are off (default 0.25) |
| `RANDOM_STATE` | Reproducibility seed (default 42) |
| `SHUFFLE_LABELS` | `1` for the shuffled-label control experiment |
| `USE_CLASS_WEIGHTS` | `1` to weight classes by inverse frequency |
| `APPLY_PCA`, `PCA_N_COMPONENTS`, `PCA_EXPLAINED_VARIANCE_THRESHOLD` | Optional PCA on aggregated features |
| `MIN_SAMPLES_PER_TYPE`, `TOP_N_TUMOR_TYPES` | Tumor-type filtering (default `100` / `None`) |
| `SNV_FEATURES_DIR`, `CNA_FEATURES_DIR`, `DATA_DIR`, `CLINICAL_PATH`, `MODELS_DIR`, `PLOTS_DIR`, `RESULTS_DIR` | Path overrides |
| `MATPLOTLIB_BACKEND` | e.g. `Agg` for headless |

## Outputs

| Path | Description |
|---|---|
| `models_macro/snv_cna_ensemble_results_<SNV>_<CNA>.pkl` | Per-fold trained Keras MLPs + label encoder + ensemble OOF predictions |
| `plots/` | ROC + PR + confusion-matrix figures |

The classifier merges COAD + READ -> COADREAD and ESCA + STAD ->
ESCASTAD (TCGA convention) and filters to tumour types with at least
100 samples in the cohort with both SNV and CNA data.

## Aggregation

Per-sample features are formed by concatenating two pooled vectors:

- **SNV**: weighted-average + max pooling of variant-level features +
  `log1p(mutational_burden)` (same as the SNV-only classifier).
- **CNA**: mean + max pooling of segment-level features +
  `log1p(segment_count)` (same as the CNA-only classifier).

The two are concatenated into a single sample-level feature vector
fed into the MLP ensemble.

## Compute requirements

Trained on an NVIDIA RTX 6000 Ada.
