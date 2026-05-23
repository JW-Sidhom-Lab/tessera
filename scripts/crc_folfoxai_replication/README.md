# FOLFOXai (Abraham 2021) replication on MSK-CHORD

Faithful re-implementation of the FOLFOXai signature reported in
[Abraham et al., *Clinical Cancer Research* 27(4):1174-1183 (2021)](https://doi.org/10.1158/1078-0432.CCR-20-3286),
applied to the MSK-CHORD 1L stage IV colorectal cancer FOLFOX / FOLFIRI
cohort. Used as the published-prior-art comparator for the TESSERA
predictive-biomarker analysis (manuscript Discussion; Supplementary
Figure 13).

Abraham's signature is an ensemble of five 1,000-tree random forests
trained over a 67-gene panel of somatic mutations and copy-number calls,
returning a binary increased-benefit (IB) / decreased-benefit (DB)
classification with a 3% no-call buffer around `p_IB = 0.5`. IB / DB
labels in the original paper are defined by time-to-next-treatment
discontinuation (TTNTD) at a 270-day cut-off.

## Inputs

| Path | Source |
|---|---|
| `abraham67_genes.txt` | Plain-text list of the 67 Abraham gene symbols, vendored from the header of the Abraham 2021 supplementary CSV. The full per-patient supplementary CSV is not redistributed; only the gene list is needed for the replication. |
| `../../data/msk_chord_2024/GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv` | MSK-CHORD 1L stage IV CRC clinical ground truth. See [`data/msk_chord_2024/README.md`](../../data/msk_chord_2024/README.md) for the MSK-CHORD download path (cBioPortal study `msk_chord_2024`). |
| `../data/msk_chord/snv.csv` | MSK-CHORD somatic mutation table (MSK-IMPACT panel calls), produced by the MSK-CHORD SNV data-prep pipeline. |
| `../data/msk_chord/cna_panel_filtered.csv` | MSK-CHORD panel-filtered copy-number segment table, produced by the MSK-CHORD CNA data-prep pipeline. |

## Layout

| Script | Role |
|---|---|
| `00_gene_overlap.py` | Diagnostic. Computes the Abraham 67 ∩ MSK-IMPACT intersection and writes the full overlap table and a missing-genes-only subset. |
| `1_build_features.py` | Builds the per-patient Abraham-67 feature matrix and the IB / DB label on the MSK-CHORD 1L FOLFOX / FOLFIRI cohort. |
| `2_train_rf_ensemble.py` | Trains the Abraham-faithful 5-config × 1,000-tree random-forest ensemble, computes the ensemble-mean `p_IB`, applies the 3% no-call buffer, and scores FOLFIRI patients out-of-sample. |
| `3_plot_panels.py` | Renders the Abraham-paper-mirror panels (per-arm KM by predicted class plus the forest of per-arm class HRs) and the manuscript-Figure-6-analogue panels (pooled IB vs DB; within-stratum FOLFOX vs FOLFIRI) on PFS and OS, including the composite 2x3 grid. |
| `_plot_style.py` | Shared Nature-style matplotlib helpers (`apply_nature_style`, `save_panel`, `FIG_W_IN`, `FIG_H_IN`, `DPI`). |

## Run order

```bash
cd scripts/crc_folfoxai_replication
python3 00_gene_overlap.py        # diagnostic, ~1 s
python3 1_build_features.py       # ~30 s
python3 2_train_rf_ensemble.py    # 5 configs x 5 folds, ~3-5 min on CPU
python3 3_plot_panels.py          # renders all panels, ~15 s
```

Each script honours env-var overrides for the input and output paths
(`ABRAHAM_GENES_TXT`, `GT_CSV`, `SNV_CSV`, `CNA_CSV`, `OUTPUT_DIR`,
`OUTPUT_CSV`).

## Outputs

```
outputs/
├── abraham67_msk_overlap.csv               # from 00_gene_overlap.py: per-gene overlap (67 rows; in_msk_snv, in_msk_cna, in_msk_impact)
└── abraham67_msk_missing.csv               # from 00_gene_overlap.py: the 39 Abraham genes not on MSK-IMPACT

causal_inference_results/
├── features/
│   └── patient_features.csv                # from 1_build_features.py
├── predictions/
│   ├── folfoxai_predictions.csv            # from 2_train_rf_ensemble.py
│   └── rf_training_log.csv                 # OOF fold-by-fold AUC log
└── paper_figures/
    ├── folfoxai_histogram/                 # Abraham-mirror: p_IB histogram
    ├── folfoxai_km_folfox/                 # Abraham-mirror: FOLFOX-arm KM by predicted class
    ├── folfoxai_km_folfiri/                # Abraham-mirror: FOLFIRI-arm KM by predicted class
    ├── folfoxai_forest/                    # Abraham-mirror: per-arm class HR forest
    ├── folfoxai_pooled_prognostic_pfs/     # Manuscript Fig 6c analogue, PFS
    ├── folfoxai_pooled_prognostic_os/      # Manuscript Fig 6c analogue, OS
    ├── folfoxai_ib_stratum_by_arm_pfs/     # Manuscript Fig 6d analogue, PFS
    ├── folfoxai_ib_stratum_by_arm_os/      # Manuscript Fig 6d analogue, OS
    ├── folfoxai_db_stratum_by_arm_pfs/     # Manuscript Fig 6e analogue, PFS
    ├── folfoxai_db_stratum_by_arm_os/      # Manuscript Fig 6e analogue, OS
    └── folfoxai_fig6_grid/                 # Composite 2x3 of the six manuscript Fig 6 c/d/e analogues
```

Each panel directory contains a `panel.png` (600 dpi), a vector
`panel.pdf`, and a `km_stats.csv` (or `histogram_stats.csv` /
`interaction_stats.csv` / `forest_rows.csv` for the non-KM panels).

## Reproducibility

- `RandomForestClassifier` random states are fixed at `[0, 1, 2, 3, 4]`
  for the five ensemble configurations; `StratifiedKFold` seed is fixed
  at `42` for the FOLFOX 5-fold out-of-fold split.
- Per-patient deduplication in `1_build_features.py` uses
  `sort_values('OS_MONTHS').drop_duplicates('PATIENT_ID', keep='first')`,
  the earliest survival row per patient.
- PFS and OS columns used by the manuscript-Fig-6-analogue panels are
  re-derived inside `3_plot_panels.py` directly from the MSK-CHORD
  ground-truth table, capped administratively at 36 months (PFS) and
  60 months (OS) so that endpoint construction matches the rest of the
  TESSERA pipeline.
- All random forests run on CPU; no GPU is required.

## Caveats

- **Abraham 67 ∩ MSK-IMPACT = 28 genes.** The Abraham signature was
  originally trained on the broader Caris Molecular Intelligence panel.
  MSK-IMPACT does not profile the remaining 39 genes, enumerated in
  `outputs/abraham67_msk_missing.csv`. The replication therefore tests
  the MSK-IMPACT-overlap portion of the FOLFOXai feature set, not the
  full 67-gene signature.
- **3% no-call buffer.** Abraham's binary classifier abstains on
  patients with `0.47 < p_IB < 0.53`. On this cohort that buffer
  excludes roughly 23% of patients from the IB / DB call; panels and
  the head-to-head comparisons restrict to called patients only.
- **Genotype filter.** Patients with no Abraham-gene SNV and an
  all-diploid Abraham-gene CNA profile are dropped during feature
  construction; this typically affects a small number of patients
  (single digits) beyond the no-call exclusion.
- **TTNTD-based IB / DB label.** This matches the Abraham 2021 paper
  but is a coarsening of progression-free survival; the manuscript
  Figure-6-analogue comparisons additionally evaluate the score against
  PFS and OS as the survival endpoints, with no relabelling of IB / DB.

## Citations

- Abraham JP et al. *Clinical Validation of a Machine-learning-derived Signature Predictive of Outcomes from First-line Oxaliplatin-based Chemotherapy in Advanced Colorectal Cancer.* Clin Cancer Res. 2021;27(4):1174-1183. doi:10.1158/1078-0432.CCR-20-3286.
- Jee J et al. *Automated real-world data integration improves cancer outcome prediction.* Nature. 2024 (MSK-CHORD cohort).
