# CNA-only tumor-type classification (Figure 3)

Trains an ensemble of MLP tumor-type classifiers on top of TESSERA CNA
features, evaluated under macro-fold nested cross-validation. Reports
per-class ROC + PR curves and the cross-model comparison panels in
Figure 3.

## Pipeline

```
data/TCGA_PanCan/clinical.csv                    (patient -> tumour type)
scripts/tcga_pancan_cna/cna_features/            (TESSERA per-segment features)
                       │
                       ▼
              tumor_type_classifier_cna.py    (one MODEL at a time)
                       │
                       ▼
              models_macro/cna_ensemble_results_<MODEL>.pkl
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
plot_from_results.py            compare_models.py
(per-model figures)             (cross-model comparison)
       │                               │
       ▼                               ▼
       plots/
```

## Configurations

The 3 attention-bearing CNA models from
[`scripts/tcga_pancan_cna/`](../tcga_pancan_cna/README.md) (`attn_0`,
`attn_1`, `attn_2`) are each evaluated as a separate classifier.
Default training mode is **macro-fold nested CV** (5 outer / 10 inner
folds) so each sample is tested exactly once.

## Running

```bash
# Train all 3 classifiers (one per CNA model)
./run_cna_classifier.sh

# Or one at a time
MODEL_NAME=attn_2 python tumor_type_classifier_cna.py
MODEL_NAME=attn_0 python tumor_type_classifier_cna.py

# Per-model figures
MODEL_NAME=attn_2 python plot_from_results.py

# Cross-model comparison
python compare_models.py
```

The classifier reads:

- per-segment TESSERA features from
  [`scripts/tcga_pancan_cna/cna_features/`](../tcga_pancan_cna/README.md)
  (produced by `get_cna_features.py`)
- patient-level tumour types from
  [`data/TCGA_PanCan/clinical.csv`](../../data/TCGA_PanCan/README.md)

so the upstream CNA pretraining + feature extraction must have run
first, and the TCGA clinical table must be present.

### Recognised env vars

| Variable | Notes |
|---|---|
| `MODEL_NAME` | CNA model whose features to use (default `attn_2`) |
| `N_FOLDS`, `N_MACRO_FOLDS`, `N_INNER_FOLDS` | CV depth (10 / 5 / 10 by default) |
| `USE_MACRO_NESTED_CV`, `USE_FULL_DATASET_CV` | CV mode toggles |
| `TEST_SIZE` | Hold-out fraction when both CV modes are off (default 0.25) |
| `RANDOM_STATE` | Reproducibility seed (default 42) |
| `SHUFFLE_LABELS` | `1` for the shuffled-label control experiment |
| `USE_CLASS_WEIGHTS` | `1` to weight classes by inverse frequency |
| `APPLY_PCA`, `PCA_N_COMPONENTS`, `PCA_EXPLAINED_VARIANCE_THRESHOLD` | Optional PCA on aggregated features |
| `MIN_SAMPLES_PER_TYPE`, `TOP_N_TUMOR_TYPES` | Tumor-type filtering (default `100` / `None`) |
| `FEATURES_DIR`, `CLINICAL_PATH`, `MODELS_DIR`, `PLOTS_DIR`, `RESULTS_DIR` | Path overrides |
| `MATPLOTLIB_BACKEND` | e.g. `Agg` for headless |

## Outputs

| Path | Description |
|---|---|
| `models_macro/cna_ensemble_results_<MODEL>.pkl` | Per-fold trained Keras MLPs + label encoder + ensemble OOF predictions |
| `plots/` | ROC + PR + confusion-matrix figures (per-model and cross-model) |

The classifier merges COAD + READ -> COADREAD and ESCA + STAD ->
ESCASTAD (TCGA convention) and filters to tumour types with at least
100 samples in the CNA dataset.

## Aggregation

Per-segment 144-dim CNA features are pooled to a sample-level vector by
mean + max pooling, with `log1p(segment_count)` appended (289 features
per sample). Unlike the SNV classifier, no VAF weighting is applied.

## Compute requirements

Trained on an NVIDIA RTX 6000 Ada.
