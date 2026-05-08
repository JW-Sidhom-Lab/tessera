# UCEC clinical metadata (TCGA PanCancer Atlas molecular subtypes)

Per-patient TCGA Pan-Cancer Atlas molecular subtype labels plus
TCGA Pan-Cancer Clinical Data Resource survival endpoints for the
TCGA-UCEC cohort (uterine corpus endometrial carcinoma). Used by
Figure 5 b (unsupervised UCEC UMAP) and the Methods §"Unsupervised
prognostic biomarkers" subtype-recovery analysis.

## Inputs

| File | Source |
|---|---|
| `ucec_clinical.tsv` | [Kandoth et al. 2013, *Nature* **497**, 67-73](https://doi.org/10.1038/nature12113) (original four-class molecular subtyping) with the [Hoadley et al. 2018, *Cell* **173**, 291-304.e6](https://doi.org/10.1016/j.cell.2018.03.022) PanCancer Atlas reannotation. Per-patient clinical-attributes table for cBioPortal's [Uterine Corpus Endometrial Carcinoma (TCGA, PanCancer Atlas)](https://www.cbioportal.org/study/clinicalData?id=ucec_tcga_pan_can_atlas_2018) study (`ucec_tcga_pan_can_atlas_2018`). The relevant column is `Subtype`, carrying the four labels `UCEC_CN_HIGH`, `UCEC_CN_LOW`, `UCEC_MSI`, and `Copy-number low (Endometriod)`. Histologic grade is in the `Grade.1` column (the second of two `Grade` columns in cBioPortal's wide-format export). To regenerate the TSV: open the study page above, switch to the "Clinical Data" tab, and click "Download". The same data is also available at [`cBioPortal/datahub` on GitHub](https://github.com/cBioPortal/datahub/tree/master/public/ucec_tcga_pan_can_atlas_2018) as `data_clinical_patient.txt` (git-lfs; clone with `git lfs install && git clone https://github.com/cBioPortal/datahub` then `git lfs pull -I 'public/ucec_tcga_pan_can_atlas_2018/*'`), and programmatically via [`https://www.cbioportal.org/api/studies/ucec_tcga_pan_can_atlas_2018/clinical-data`](https://www.cbioportal.org/api/studies/ucec_tcga_pan_can_atlas_2018/clinical-data). |
| `../../TCGA_PanCan/clinical.csv` | [Liu et al. 2018, *Cell* **173**, 400-416.e11](https://doi.org/10.1016/j.cell.2018.02.052). Curated DSS / DFI / PFI endpoints from the TCGA Pan-Cancer Clinical Data Resource. |

## Processing

`build_ucec_metadata.py`:

1. Load `ucec_clinical.tsv`; keep `Patient ID`, `Sample ID`,
   `Cancer Type`, `Cancer Type Detailed`, `Grade.1` (renamed to
   `Grade`), and `Subtype`; deduplicate to one row per patient.
2. Load `clinical.csv`, restrict to `type=='UCEC'`; remap each curated
   event code `2 -> 0` (Liu 2018 ambiguous-cause -> censored); drop
   the `.cr` / `_cr` suffix when writing the columns
   (`DSS_cr` -> `DSS`, `DSS.time.cr` -> `DSS.time`, and analogously for
   `DFI`, `PFI`).
3. Inner-join the subtype and survival tables on `Patient_ID` and
   write `ucec_clinical_metadata.csv`.

```bash
cd data/prognostic/ucec
python build_ucec_metadata.py
```

Path overrides: `SUBTYPE_TSV`, `CLINICAL_CSV`, `OUTPUT_CSV`.

## Output

`ucec_clinical_metadata.csv`, one row per UCEC patient (n = 460):

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode (first 12 chars). |
| `Sample ID`, `Cancer Type`, `Cancer Type Detailed` | cBioPortal-derived identifiers. |
| `Grade` | Histologic grade (G1 / G2 / G3 and pre-2018 string variants). |
| `Subtype` | TCGA PanCancer molecular subtype: `UCEC_CN_HIGH` (copy-number high, serous-like), `UCEC_CN_LOW` (copy-number low, endometrioid), `UCEC_MSI` (microsatellite-unstable hypermutated), `Copy-number low (Endometriod)` (original Kandoth 2013 label retained for one subset). |
| `DSS`, `DSS.time` | Disease-specific survival event (binary) and time in days (Liu 2018 curated). |
| `DFI`, `DFI.time` | Disease-free interval. |
| `PFI`, `PFI.time` | Progression-free interval. |

## Caveats

- The `Subtype` column carries four labels rather than the three
  originally proposed by Kandoth 2013, because the PanCancer Atlas
  reannotation redrew the copy-number-low boundary. Both labellings
  are retained verbatim from cBioPortal; downstream analyses keep all
  four classes.
- The cBioPortal `Molecular Subtype` column (alphabetically distinct
  from `Subtype`) is empty for this study and is not used.
- `clinical.csv` does not carry overall-survival columns (`OS`,
  `OS.time`); only DSS / DFI / PFI are joined. Methods §240 uses DSS
  for UCEC, consistent with the BRCA setup.

## Citations

- Kandoth et al. *Integrated genomic characterization of endometrial carcinoma*. *Nature* **497**, 67-73 (2013).
- Hoadley et al. *Cell-of-origin patterns dominate the molecular classification of 10,000 tumors from 33 types of cancer*. *Cell* **173**, 291-304.e6 (2018).
- Liu et al. *An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics*. *Cell* **173**, 400-416.e11 (2018).
