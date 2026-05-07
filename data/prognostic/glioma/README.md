# Glioma clinical metadata (WHO 2021 reclassification)

Per-patient WHO 2021 primary class plus TCGA Pan-Cancer Atlas survival
endpoints for the TCGA glioma cohort. Used by Figure 4 g-h
(classifier-vs-pathology concordance under WHO 2021) and Figure 5 a
(unsupervised glioma UMAP).

## Inputs

| File | Source |
|---|---|
| `Matrix_WHO2021.csv` | Leiria et al. 2025, *Scientific Data* **12**, 5117. Per-patient WHO 2021 simplified primary class. |
| `../../TCGA_PanCan/clinical.csv` | Liu et al. 2018, *Cell* **173**, 400-416.e11. Curated DSS / DFI / PFI endpoints. |

## Processing

`build_glioma_metadata.py`:

1. Load `Matrix_WHO2021.csv`; keep `Patient_ID` and the simplified
   WHO 2021 label (`glioblastoma` / `astrocytoma` /
   `oligodendroglioma` / `unclassified`); deduplicate.
2. Load `clinical.csv`; remap each curated event code `2 -> 0` (Liu
   2018 ambiguous-cause-of-death -> censored); drop the `.cr` /
   `_cr` suffix when writing the columns (`DSS_cr` -> `DSS`,
   `DSS.time.cr` -> `DSS.time`, and analogously for `DFI`, `PFI`).
3. Inner-join on `Patient_ID` and write
   `glioma_clinical_metadata.csv`.

```bash
cd data/prognostic/glioma
python build_glioma_metadata.py
```

Path overrides: `WHO2021_CSV`, `CLINICAL_CSV`, `OUTPUT_CSV`.

## Output

`glioma_clinical_metadata.csv`, one row per glioma patient:

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode (first 12 chars). |
| `WHO2021` | Simplified WHO 2021 primary class. |
| `DSS`, `DSS.time` | Disease-specific survival event (binary) and time in days (Liu 2018 curated). |
| `DFI`, `DFI.time` | Disease-free interval. |
| `PFI`, `PFI.time` | Progression-free interval. |
