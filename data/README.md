# Data preparation

Each subdirectory holds the data-preparation pipeline for one of the public
cohorts used in the TESSERA manuscript. Raw release files and derived data
products are gitignored; users download them from the canonical source and
run the cohort's `create_training_data.py` (or equivalent) to regenerate the
training-ready CSVs.

| Cohort | Manuscript role | Source |
|---|---|---|
| [`TCGA_PanCan/`](TCGA_PanCan/README.md) | Pretraining corpus (SNV + CNA) | TCGA Pan-Cancer Atlas, GDC PanCanAtlas page |
| [`genie_18_0/`](genie_18_0/README.md) | SNV cross-platform validation (Fig. 1 f-g) | AACR Project GENIE v18.0, Synapse `syn7222066` |
| [`clinvar/`](clinvar/README.md) | Variant-pathogenicity labels (Fig. 1 h-o) | NCBI ClinVar, GRCh37 VCF release |
| [`msk_chord_2024/`](msk_chord_2024/README.md) | CNA cross-platform validation (Fig. 2 d) and clinical predictive-biomarker analyses (Fig. 6 b-i) | MSK-CHORD 2024, cBioPortal / Synapse |
| [`DepMap/`](DepMap/README.md) | CRC cell-line orthogonal validation (Fig. 6 n) | DepMap 24Q2 + CTRPv2 |

See each cohort's `README.md` for the download URL, required files, run
commands, and expected outputs.

## Conventions

- Each cohort directory contains its raw inputs (gitignored), one or more
  `create_training_data*.py` scripts (or `prepare_data.py` + cohort builders
  for MSK-CHORD), and the derived CSVs (gitignored).
- Scripts read defaults from the current working directory and accept
  CLI arg overrides (`--maf`, `--clinical`, `--output`, etc.). Run from
  inside the cohort directory to use the defaults.
- All scripts are import-safe (no module-level side effects) and emit
  per-stage row counts via Python `logging` at INFO level.

## Dependencies

The data-preparation scripts require `pandas`, `numpy`, and (for the TCGA
clinical xlsx, if you regenerate it) `openpyxl`. No GPU. Memory peaks around
35-40 GB during the TCGA MC3 MAF load; the others fit comfortably in 8 GB.
