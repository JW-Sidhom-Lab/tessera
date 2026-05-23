# Analysis pipelines

Each subdirectory holds the pipeline for one stage of the TESSERA
manuscript: data preparation, foundation-model pretraining, or a
downstream analysis backing one or more published figures. Cohort
data prep ([`data/`](data/README.md)) and pretrained model weights
are gitignored; users regenerate the data tables from the canonical
sources (see [`../data/`](../data/README.md)) and run the pretraining
scripts to recreate the model checkpoints, after which the downstream
scripts run end-to-end against the pretrained foundation model.

## Pipeline

```
../data/<cohort>/                           (raw releases + clinical metadata, downloaded by the user)
        │
        ▼ scripts/data/<cohort>/            (model-ready train/valid CSVs, snv/cna tables)
        │
        ▼ scripts/{tcga_pancan_snv,tcga_pancan_cna,tcga_pancan_cna_noloh,tcga_pancan_snv_cna}/
        │   (TCGA pretraining: SNV, CNA, joint SNV+CNA InfoNCE)
        │
        ▼ scripts/{genie_snv,msk_chord_cna}/
        │   (cross-platform validation of the pretrained models)
        │
        ▼ scripts/variant_effect/, scripts/tumor_classification_*/
        │   (downstream supervised tasks built on frozen TESSERA features)
        │
        ▼ scripts/{prognostic_bm,predictive_bm,depmap,crc_folfoxai_replication}/
            (clinical biomarker analyses)
```

## Subdirectories

### Data preparation

| Directory | Manuscript role |
|---|---|
| [`data/`](data/README.md) | Per-cohort Stage-2 data preparation: builds the model-ready SNV / CNA CSVs that the pretraining and validation scripts consume. One subdirectory per cohort (`tcga/`, `genie/`, `msk_chord/`, `depmap/`). |

### Foundation-model pretraining

| Directory | Manuscript role |
|---|---|
| [`tcga_pancan_snv/`](tcga_pancan_snv/README.md) | Trains the seven TESSERA SNV models (baseline + local_{1, 10, 25} + global_{1, 10, 25}) on TCGA Pan-Cancer SNVs. Backs Fig. 1 c-e + Sup Fig 1. |
| [`tcga_pancan_cna/`](tcga_pancan_cna/README.md) | Trains the three TESSERA CNA models (no-attention baseline + 1-block + 2-block self-attention) on TCGA Pan-Cancer copy-number segments. Backs Fig. 2. |
| [`tcga_pancan_cna_noloh/`](tcga_pancan_cna_noloh/README.md) | NoLOH variant of `tcga_pancan_cna/`: drops the LoH input head to match the MSK-IMPACT panel (no LoH calls). Produces the CNA encoder used by the joint SNV+CNA pipeline. |
| [`tcga_pancan_snv_cna/`](tcga_pancan_snv_cna/README.md) | Joint SNV+CNA InfoNCE-aligned pretraining; the multimodal foundation model used by every downstream clinical analysis. |

### Cross-platform validation

| Directory | Manuscript role |
|---|---|
| [`genie_snv/`](genie_snv/README.md) | TCGA-trained SNV models applied to AACR Project GENIE v18.0 panel-sequencing data. Backs Fig. 1 f-g. |
| [`variant_effect/`](variant_effect/README.md) | TCGA-trained SNV models applied to ClinVar for variant pathogenicity prediction (variant-level + gene-level deduplication). Backs Fig. 1 h-o + Sup Fig 2. |
| [`msk_chord_cna/`](msk_chord_cna/README.md) | TCGA-trained CNA models applied to the MSK-CHORD MSK-IMPACT panel-segmentation cohort. Backs Fig. 2 d. |

### Tumour-type classification

| Directory | Manuscript role |
|---|---|
| [`tumor_classification_snv/`](tumor_classification_snv/README.md) | MLP classifier ensemble on frozen SNV features. Backs Fig. 3 b-c + Sup Fig 3-5. |
| [`tumor_classification_cna/`](tumor_classification_cna/README.md) | MLP classifier ensemble on frozen CNA features. Backs Fig. 3 d-e + Sup Fig 6-8. |
| [`tumor_classification_snv_cna/`](tumor_classification_snv_cna/README.md) | MLP classifier ensemble on concatenated SNV + CNA features. Joint-modality baseline used as comparator for the InfoNCE-aligned classifier. |
| [`tumor_classification_snv_cna_infonce/`](tumor_classification_snv_cna_infonce/README.md) | MLP classifier ensemble on the joint InfoNCE-aligned per-sample features. Backs Fig. 4 b-e + Sup Fig 9. |

### Clinical biomarker analyses

| Directory | Manuscript role |
|---|---|
| [`prognostic_bm/`](prognostic_bm/README.md) | Per-cohort UMAP + polynomial Cox + risk-group Kaplan-Meier on the joint InfoNCE-aligned per-sample features. Backs Fig. 4 g-h (glioma classifier concordance) and Fig. 5 a-r (glioma + BRCA + PRAD). |
| [`predictive_bm/`](predictive_bm/README.md) | Doubly-robust counterfactual treatment-effect pipeline (CRC FOLFOX vs FOLFIRI; PDAC FOLFIRINOX vs gemcitabine + nab-paclitaxel) + PMD attribution + genotype-rule subgroups. Backs Fig. 6 a-m + Sup Fig 10-12. |
| [`depmap/`](depmap/README.md) | Cross-cohort transfer of the MSK-CHORD-trained CRC predictive biomarker to DepMap colorectal cell lines, tested against CTRPv2 oxaliplatin and SN-38 dose-response. Backs Fig. 6 n. |
| [`crc_folfoxai_replication/`](crc_folfoxai_replication/README.md) | Faithful re-implementation of the published FOLFOXai signature (Abraham 2021) on the same MSK-CHORD 1L stage IV CRC FOLFOX / FOLFIRI cohort; head-to-head comparator showing the FOLFOXai-on-MSK-CHORD score is prognostic-only without the arm-stratified direction reversal that defines a predictive biomarker under the Ballman 2015 framework. Backs Sup Fig 13. |

## Conventions

- Each analysis directory contains a `README.md` with the cohort,
  manuscript figure(s) it backs, the input + output paths, and the run
  commands.
- Inference and analysis scripts use top-level constants + env-var
  overrides rather than `argparse` (`argparse` breaks IPython /
  Jupyter / REPL debugging). The Stage-2 data-prep scripts under
  [`data/`](data/README.md) keep `argparse` because they are batch
  jobs invoked from shell drivers.
- Path constants are repo-relative (`../data/<cohort>/...`,
  `../tcga_pancan_snv_cna/models/...`, `../predictive_bm/cache/...`),
  so the layout transfers without edits.
- Per-script artefact directories (`models/`, `var_features/`,
  `var_loss/`, `multimodal_features/`, `attribution_analysis/`,
  `cache/`, `results/`, `plots/`, etc.) and the large `.pkl`
  inference outputs are gitignored.
- Pretrained model weights are not included; users regenerate them by
  running the corresponding pretraining script. The clinical
  biomarker analyses depend on the joint SNV+CNA InfoNCE-noLOH model
  produced by [`tcga_pancan_snv_cna/`](tcga_pancan_snv_cna/README.md).

## Dependencies

`tensorflow`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `lifelines`,
`matplotlib`, `seaborn`. Cohort-specific extras:

- `pyreadr` for `data/prognostic/prad/build_prad_metadata.py`
- GPU + `tensorflow-gpu` (or equivalent) for the pretraining and
  frozen-model inference scripts; CPU-only is sufficient for the
  downstream survival, classifier, and biomarker analyses.
