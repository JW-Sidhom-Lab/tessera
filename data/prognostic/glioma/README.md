# Glioma clinical metadata (WHO 2021 reclassification)

Per-patient WHO 2021 primary class plus TCGA Pan-Cancer Atlas survival
endpoints for the TCGA glioma cohort (GBM + LGG, n=1,110). Used by the
manuscript's classifier-vs-pathology concordance analyses (Figure 4 g-h)
and the unsupervised glioma UMAP (Figure 5 a) to test alignment with
the post-2021 molecular classification that retired the historical
GBM/LGG histologic split.

## Inputs

| File | Source | Citation |
|---|---|---|
| `Matrix_WHO2021.csv` | Leiria et al. 2025 supplementary table | Leiria, R. *et al.* Updated TCGA glioma classification according to the 2021 WHO classification of CNS tumours. *Scientific Data* **12**, 5117 (2025). doi: 10.1038/s41597-025-05117-2 |
| `../../TCGA_PanCan/clinical.csv` | TCGA Pan-Cancer Atlas curated clinical resource (Liu 2018-style) | Liu, J. *et al.* An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. *Cell* **173**, 400-416.e11 (2018). |

`Matrix_WHO2021.csv` was downloaded from the supplementary materials of
Leiria 2025 and committed verbatim. The `classification.2021_simplified.labels`
column collapses the full WHO 2021 entity vocabulary into four primary
classes used throughout the manuscript: **glioblastoma**, **astrocytoma**,
**oligodendroglioma**, and **unclassified**.

`clinical.csv` is the Liu 2018 curated TCGA Pan-Cancer Atlas clinical
resource, gitignored as a raw upstream input under
[`data/TCGA_PanCan/`](../../TCGA_PanCan/README.md). The build reads the
three curated endpoint pairs
(`DSS_cr`/`DSS.time.cr`, `DFI.cr`/`DFI.time.cr`,
`PFI.cr`/`PFI.time.cr`) and the `bcr_patient_barcode`.

## Processing

`build_glioma_metadata.py`:

1. Loads `Matrix_WHO2021.csv`, keeps `Patient_ID` and the simplified
   WHO 2021 label, deduplicates to one row per patient.
2. Loads `clinical.csv`, derives `Patient_ID` from the first 12
   characters of `bcr_patient_barcode`, deduplicates, remaps the
   ambiguous-cause event code `2` to `0` (censored) on each curated
   endpoint per the Liu 2018 convention, casts events to `Int64`, and
   renames the three `_cr`-suffixed pairs to the legacy column names
   `DSS`/`DSS.time`, `DFI`/`DFI.time`, `PFI`/`PFI.time`.
3. Inner-joins on `Patient_ID` and writes
   `glioma_clinical_metadata.csv`.

Run:

```bash
cd data/prognostic/glioma
python build_glioma_metadata.py
```

Override paths via the `WHO2021_CSV`, `NCIT_CSV`, and `OUTPUT_CSV`
environment variables.

## Output

`glioma_clinical_metadata.csv` -- one row per glioma patient, columns:

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode (first 12 chars). |
| `WHO2021` | Simplified WHO 2021 primary class. |
| `DSS`, `DSS.time` | Liu 2018 curated disease-specific survival (`DSS_cr`/`DSS.time.cr` with `2 -> 0` remap). |
| `DFI`, `DFI.time` | Liu 2018 curated disease-free interval. |
| `PFI`, `PFI.time` | Liu 2018 curated progression-free interval. |

Of the 1,110 patients, 1,109 carry `DSS.time`.

OS is intentionally not emitted: `clinical.csv` only ships Liu 2018
curated DSS / DFI / PFI columns. None of the manuscript Figure 4 or
Figure 5 analyses use OS for glioma.
