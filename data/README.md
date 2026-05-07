# Data preparation

Each subdirectory holds the data-preparation pipeline for one of the public
cohorts used in the TESSERA manuscript. Users download the raw release files
from the canonical source and run the cohort's build script to regenerate
the analysis-ready CSVs.

| Cohort | Manuscript role | Source |
|---|---|---|
| [`TCGA_PanCan/`](TCGA_PanCan/README.md) | Pretraining corpus (SNV + CNA) | TCGA Pan-Cancer Atlas, GDC PanCanAtlas page |
| [`genie_18_0/`](genie_18_0/README.md) | SNV cross-platform validation (Fig. 1 f-g) | AACR Project GENIE v18.0, Synapse `syn7222066` |
| [`clinvar/`](clinvar/README.md) | Variant-pathogenicity labels (Fig. 1 h-o) | NCBI ClinVar, GRCh37 VCF release |
| [`msk_chord_2024/`](msk_chord_2024/README.md) | CNA cross-platform validation (Fig. 2 d) and clinical predictive-biomarker analyses (Fig. 6 b-i) | MSK-CHORD 2024, cBioPortal / Synapse |
| [`DepMap/`](DepMap/README.md) | CRC cell-line orthogonal validation (Fig. 6 n) | DepMap 24Q2 + CTRPv2 |
| [`prognostic/`](prognostic/README.md) | Per-cohort clinical metadata + published transcriptomic comparators for the glioma WHO 2021 classifier concordance (Fig. 4 g-h) and the unsupervised prognostic biomarker UMAPs + joint Cox (Fig. 5 a-r) | Liu 2018 (TCGA Pan-Cancer Clinical Data Resource), Howard 2023 (DLRS), Laajala 2023 (curatedPCaData), Leiria 2025 (MONET) |

See each cohort's `README.md` for the download URL, required files, run
commands, and expected outputs.

## Conventions

Two pipeline shapes coexist:

- **Variant / segment cohorts** (`TCGA_PanCan/`, `genie_18_0/`, `clinvar/`,
  `msk_chord_2024/`, `DepMap/`) contain one or more `create_training_data*.py`
  scripts (or `prepare_data.py` + cohort builders for `msk_chord_2024/`).
  Raw inputs and derived CSVs are gitignored; the scripts accept CLI arg
  overrides (`--maf`, `--clinical`, `--output`, etc.), are import-safe (no
  module-level side effects), and emit per-stage row counts via Python
  `logging` at INFO level.
- **Clinical-metadata cohorts** (`prognostic/glioma/`, `prognostic/brca/`,
  `prognostic/prad/`) include the small published source tables they
  consume, a `build_<cohort>_metadata.py` builder, and the resulting
  `<cohort>_clinical_metadata.csv`. The builders run at module load and
  accept env-var overrides (`WHO2021_CSV`, `DLRS_CSV`, `SCORES_RDS`,
  `CLINICAL_CSV`, `OUTPUT_CSV`).

In both cases, run the scripts from inside the cohort directory to use the
defaults.

## Dependencies

`pandas` and `numpy` are required everywhere. Cohort-specific extras:

- `openpyxl` for the TCGA clinical xlsx (only if you regenerate it)
- `pyreadr` for `prognostic/prad/build_prad_metadata.py` (reads the
  curatedPCaData `.Rds` archive)

No GPU. Memory peaks at ~35-40 GB during the TCGA MC3 MAF load in
`TCGA_PanCan/`; every other cohort fits comfortably in 8 GB.
