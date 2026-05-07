# SNV+CNA InfoNCE-aligned tumor-type classification (Figure 4)

Trains an ensemble of MLP tumor-type classifiers on per-sample features
from the joint InfoNCE-aligned TESSERA model in
[`scripts/tcga_pancan_snv_cna/`](../tcga_pancan_snv_cna/README.md), then
runs the classifier-vs-pathology concordance survival analyses that
back Figure 4 d-h. Reports the multimodal-classifier panel of Figure 4
b-c and the disease-specific-survival concordance panels (d-h).

## Pipeline

```
data/TCGA_PanCan/clinical.csv                            (TCGA CDR survival)
data/TCGA_PanCan/glioma_who2021_metadata.csv             (WHO 2021 glioma labels)
scripts/tcga_pancan_snv_cna/multimodal_features/         (joint InfoNCE features)
                       │
                       ▼
              tumor_type_classifier_snv_cna_infonce.py     (per variant)
                       │
                       ▼
              models_macro/snv_cna_infonce_ensemble_results.pkl
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
plot_from_results  tcga_concordance_  tcga_concordance_
   (per-model        survival.py        paper_figures.py
    Fig 4 b-c        (Fig 4 d-e        (Fig 4 d-h panels
    figures)         pan-cancer KM     in publication
                     + per-class       layout)
                     forest)
```

## Manuscript variants

Both joint pretraining variants from `scripts/tcga_pancan_snv_cna/` are
classified independently:

| Variant | Features | Output dirs | Manuscript role |
|---|---|---|---|
| **LOH** (default) | `TCGA_SNV_CNA_InfoNCE_per_sample_loss_multimodal_features.pkl` | `models_macro/`, `plots/` | **Figure 4 b-c headline** (macro-AUC 0.987, micro-AUC 0.991, macro-AP 0.893, micro-AP 0.908) |
| NoLOH | `TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH_multimodal_features.pkl` | `models_macro_noloh/`, `plots_noloh/` | Ablation; corresponds to the noLOH joint pretraining variant |

Default training mode is **macro-fold nested CV** (5 outer / 10 inner)
so each sample is tested exactly once.

## Running

```bash
# Train both variants (manuscript)
./run_classifier.sh

# One variant only
./run_classifier.sh loh
./run_classifier.sh noloh

# Or call the classifier directly with full env-var control
FEATURES_PATH=../tcga_pancan_snv_cna/multimodal_features/TCGA_SNV_CNA_InfoNCE_per_sample_loss_multimodal_features.pkl \
    python tumor_type_classifier_snv_cna_infonce.py

# Per-variant figures
RESULTS_DIR=models_macro       PLOTS_DIR=plots       python plot_from_results.py
RESULTS_DIR=models_macro_noloh PLOTS_DIR=plots_noloh python plot_from_results.py

# Figure 4 d-e: pan-cancer + per-class concordance KM / forests
python tcga_concordance_survival.py                        # default DSS endpoint
METRIC=PFI python tcga_concordance_survival.py             # alternative endpoint

# Figure 4 d-h: paper-layout panels (DSS only)
python tcga_concordance_paper_figures.py
```

The classifier reads:

- joint per-modality features from
  [`scripts/tcga_pancan_snv_cna/multimodal_features/`](../tcga_pancan_snv_cna/README.md)
  (produced by `get_cna_variant_features.py`)
- per-patient TCGA survival from
  [`data/TCGA_PanCan/clinical.csv`](../../data/TCGA_PanCan/README.md)
  (used only by the concordance scripts)
- WHO 2021 glioma reclassification labels from
  `data/prognostic/glioma/glioma_clinical_metadata.csv`
  (used only by `tcga_concordance_paper_figures.py`)

so the upstream joint pretraining + feature extraction must have run
first.

### Recognised env vars

`tumor_type_classifier_snv_cna_infonce.py`:

| Variable | Notes |
|---|---|
| `FEATURES_PATH` | Joint-features pkl (default = LOH variant) |
| `OUTPUT_TAG` | Suffix for output dirs (default empty; set `_noloh` for NoLOH) |
| `MODELS_DIR`, `PLOTS_DIR` | Direct path overrides (override `OUTPUT_TAG`) |
| `N_FOLDS`, `N_MACRO_FOLDS`, `N_INNER_FOLDS` | CV depth (10 / 5 / 10) |
| `USE_MACRO_NESTED_CV`, `USE_FULL_DATASET_CV` | CV mode toggles |
| `TEST_SIZE` | Hold-out fraction when both above are off (0.25) |
| `RANDOM_STATE` | Reproducibility seed (42) |
| `SHUFFLE_LABELS` | `1` for the shuffled-label control |
| `USE_CLASS_WEIGHTS` | `1` to weight by inverse class frequency |
| `APPLY_PCA`, `PCA_N_COMPONENTS`, `PCA_EXPLAINED_VARIANCE_THRESHOLD` | Optional PCA |
| `MIN_SAMPLES_PER_TYPE`, `TOP_N_TUMOR_TYPES` | Tumor-type filter (100 / None) |
| `MATPLOTLIB_BACKEND` | e.g. `Agg` for headless |

`tcga_concordance_survival.py`:

| Variable | Notes |
|---|---|
| `METRIC` | `DSS` (default) / `DFI` / `PFI` / `PFS` |
| `OUTPUT_DIR` | Override (default `tcga_concordance_<METRIC>/`) |
| `MIN_TOP1_PROB`, `MIN_TRUE_CLASS_PROB` | Optional confidence thresholds |
| `RESULTS_PKL`, `CLINICAL_CSV` | Path overrides |

`tcga_concordance_paper_figures.py`:

| Variable | Notes |
|---|---|
| `RESULTS_PKL`, `CLINICAL_CSV`, `GLIOMA_META`, `OUT_ROOT` | Path overrides |

## Outputs

| Path | Description |
|---|---|
| `models_macro/`, `models_macro_noloh/` | Per-fold trained MLPs + ensemble OOF predictions for each variant |
| `plots/`, `plots_noloh/` | Per-variant ROC + PR + confusion-matrix figures |
| `tcga_concordance_DSS/` | Pan-cancer + per-class concordance survival panels (Figure 4 d-e source) |
| `tcga_concordance_results/paper_figures/` | Publication-layout concordance panels (Figure 4 d-h source) |

## Aggregation

Per-sample features mirror `tumor_classification_snv_cna/`:

- **SNV branch**: weighted-average + max pooling of per-variant features +
  `log1p(mutational_burden)`.
- **CNA branch**: mean + max pooling of per-segment features +
  `log1p(segment_count)`.
- Concatenated to a single vector before the MLP, fit per fold under
  macro-fold nested CV.

The only difference vs the concat-baseline classifier in
`tumor_classification_snv_cna/` is that the per-variant and per-segment
features come from the joint InfoNCE-aligned encoder rather than from
independently-pretrained SNV-only and CNA-only encoders.

## Compute requirements

Trained on an NVIDIA RTX 6000 Ada.
