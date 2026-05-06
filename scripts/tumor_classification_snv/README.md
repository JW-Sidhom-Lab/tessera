# SNV-only tumor-type classification (Figure 3)

Trains an ensemble of MLP tumor-type classifiers on top of TESSERA SNV
features, evaluated under macro-fold nested cross-validation. Reports
per-class ROC + PR curves and the cross-model comparison panels in
Figure 3.

## Pipeline

```
scripts/data/tcga/{train,valid}_data_snv.csv     (sample IDs + tumour types)
scripts/tcga_pancan_snv/var_features/            (TESSERA per-variant features)
                       │
                       ▼
              tumor_type_classifier_snv.py    (one MODEL at a time)
                       │
                       ▼
              models_macro/<MODEL>_results.pkl
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

The 6 attention-bearing SNV variant models from
[`scripts/tcga_pancan_snv/`](../tcga_pancan_snv/README.md) (3 local
+ 3 global context widths) are each evaluated as a separate classifier.
The no-context baseline is **not** included in this comparison; it
appears in the masked-token accuracy lineup in Figure 1 but not in the
SNV tumour-classifier panels of Figure 3 b-c. Default training mode is
**macro-fold nested CV** (5 outer / 10 inner folds) so each sample is
tested exactly once.

## Running

```bash
# Train all 6 classifiers (one per variant model)
./run_snv_classifier.sh

# Or one at a time
MODEL_NAME=global_25 python tumor_type_classifier_snv.py
MODEL_NAME=local_25  python tumor_type_classifier_snv.py

# Per-model figures
MODEL_NAME=global_25 python plot_from_results.py

# Cross-model comparison
python compare_models.py
```

The classifier reads:

- per-variant TESSERA features from
  [`scripts/tcga_pancan_snv/var_features/`](../tcga_pancan_snv/README.md)
  (produced by `get_variant_features.py`)
- variant + sample metadata from
  [`scripts/data/tcga/{train,valid}_data_snv.csv`](../data/tcga/README.md)

so the upstream SNV pretraining + feature extraction must have run
first.

### Recognised env vars

| Variable | Notes |
|---|---|
| `MODEL_NAME` | Variant model whose features to use (default `global_25`) |
| `N_FOLDS`, `N_MACRO_FOLDS`, `N_INNER_FOLDS` | CV depth (10 / 5 / 10 by default) |
| `USE_MACRO_NESTED_CV`, `USE_FULL_DATASET_CV` | CV mode toggles |
| `TEST_SIZE` | Hold-out fraction when both CV modes are off (default 0.25) |
| `RANDOM_STATE` | Reproducibility seed (default 42) |
| `SHUFFLE_LABELS` | `1` for the shuffled-label control experiment |
| `USE_CLASS_WEIGHTS` | `1` to weight classes by inverse frequency |
| `APPLY_PCA`, `PCA_N_COMPONENTS`, `PCA_EXPLAINED_VARIANCE_THRESHOLD` | Optional PCA on aggregated features |
| `MIN_SAMPLES_PER_TYPE`, `TOP_N_TUMOR_TYPES` | Tumor-type filtering (default `100` / `None`) |
| `FEATURES_DIR`, `DATA_DIR`, `MODELS_DIR`, `PLOTS_DIR`, `RESULTS_DIR` | Path overrides |
| `MATPLOTLIB_BACKEND` | e.g. `Agg` for headless |

## Outputs

| Path | Description |
|---|---|
| `models_macro/<MODEL>_results.pkl` | Per-fold trained Keras MLPs + label encoder + ensemble OOF predictions |
| `plots/` | ROC + PR + confusion-matrix figures (per-model and cross-model) |

The classifier merges COAD + READ -> COADREAD and ESCA + STAD ->
ESCASTAD (TCGA convention) and filters to tumour types with at least
100 samples in the variant dataset.

## Compute requirements

Trained on an NVIDIA RTX 6000 Ada.
