# Prognostic-biomarker cohort metadata

Per-patient clinical / outcome tables for the three TCGA cohorts in
the manuscript's prognostic-biomarker analyses (Figure 4 g-h glioma
case study; Figure 5 a-r glioma + BRCA + PRAD UMAP overlays + joint
Cox of the TESSERA risk score against published transcriptomic
comparators). One subdirectory per cohort, each self-contained:

| Cohort | Comparator | Manuscript figures | README |
|---|---|---|---|
| `glioma/` | WHO 2021 primary class ([Leiria 2025](https://doi.org/10.1038/s41597-025-05117-2)) | Figure 4 g-h, Figure 5 a | [README](glioma/README.md) |
| `brca/`   | DLRS-reconstructed OncotypeDX RS ([Howard 2023](https://doi.org/10.1038/s41523-023-00530-5)) | Figure 5 c-j | [README](brca/README.md) |
| `prad/`   | curatedPCaData Decipher score ([Laajala 2023](https://doi.org/10.1038/s41597-023-02335-4)) | Figure 5 k-r | [README](prad/README.md) |

Each subdirectory contains:

- the small published source tables it consumes,
- a `build_<cohort>_metadata.py` script that joins them with TCGA
  Pan-Cancer Atlas survival rows from
  [`../TCGA_PanCan/clinical.csv`](../TCGA_PanCan/README.md),
- the resulting `<cohort>_clinical_metadata.csv` consumed by
  `scripts/`.

## Shared conventions

- **Patient identifier**: `Patient_ID` = first 12 characters of the
  TCGA sample barcode (`bcr_patient_barcode`).
- **Survival source**: Liu 2018 curated TCGA Pan-Cancer Atlas Clinical
  Data Resource (`clinical.csv`).
- **Event encoding**: the curated `_cr` columns use a tri-state
  encoding where `2` flags ambiguous cause-of-death (uncodable). The
  build remaps `2 -> 0` (treat as censored) so the output event
  columns are strictly binary (`{0, 1}`), matching the convention
  `lifelines` expects and the convention applied by the Figure 4
  concordance scripts.
- **Endpoint columns**: glioma and BRCA emit the curated DSS / DFI /
  PFI values (the `_cr` columns from `clinical.csv` after the
  ambiguous-cause remap) under the simpler names `DSS`, `DFI`, `PFI`.
  PRAD keeps the original `_cr` column names plus the full set of Liu
  2018 endpoint variants (`PFI.1` is the recommended primary endpoint
  for PRAD because DSS is underpowered, ~5 events).
- **Reference papers** (PDFs of Leiria 2025, Howard 2023, Laajala 2023)
  are not redistributed to keep the repository copyright-clean; full
  citations live in each cohort's README and build-script docstring.
