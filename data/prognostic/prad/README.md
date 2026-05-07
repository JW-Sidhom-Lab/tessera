# PRAD clinical metadata (curatedPCaData transcriptomic comparators)

Per-patient Decipher / OncotypeDX / Prolaris / AR-signaling scores from
the published curatedPCaData TCGA release, plus the TCGA-PRAD curated
clinical / outcome columns. Used by Figure 5 k-r of the manuscript:
UMAP overlay of the TESSERA prostate-cancer manifold + joint Cox of the
TESSERA risk score against the Decipher comparator.

## Inputs

| File | Source | Citation |
|---|---|---|
| `curatedPCaData_tcga_scores_20230215.Rds` | Bioconductor ExperimentHub `EH8024` | Laajala, T. D. *et al.* A harmonized resource of integrated prostate cancer clinical, -omic, and signature features. *Scientific Data* **10**, 430 (2023). |
| `../../TCGA_PanCan/clinical.csv` | TCGA Pan-Cancer Atlas curated clinical resource | Liu, J. *et al.* *Cell* **173**, 400-416.e11 (2018). |

`curatedPCaData_tcga_scores_20230215.Rds` was downloaded from
Bioconductor ExperimentHub (resource `EH8024`) via the OSN bucket URL
`https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/curatedPCaData/tcga_scores_20230215.Rds`
and committed verbatim. It contains four per-aliquot transcriptomic
signature scores, all computed under a single uniform RNA-seq
processing pipeline:

| Score | Description |
|---|---|
| `decipher` | 22-gene metastasis classifier (Erho 2013, Karnes 2013). |
| `oncotype` | OncotypeDX GPS (17-gene risk score, Knezevic 2013). |
| `prolaris` | Cuzick CCP cell-cycle-progression score (Cuzick 2011). |
| `ar_score` | 20-gene AR-signaling activity (Hieronymus 2006). |

`clinical.csv` is the Liu 2018 curated TCGA Pan-Cancer Atlas clinical
resource, gitignored as a raw upstream input under
[`data/TCGA_PanCan/`](../../TCGA_PanCan/README.md). For PRAD we pull
all available endpoints (`PFI`, `PFI.1`, `PFI.2`, `PFS`, `DSS_cr`,
`DFI.cr`, `PFI.cr`); per Liu 2018 the recommended primary endpoint for
PRAD is `PFI.1` because OS is underpowered (~10 events).

## Processing

`build_prad_metadata.py`:

1. Loads `clinical.csv`, restricts to `type=='PRAD'` (n=500).
2. Loads `curatedPCaData_tcga_scores_20230215.Rds` via `pyreadr`,
   restricts to primary-tumor aliquots (sample-type code `01`),
   aggregates replicate aliquots per patient by mean (461 aliquots ->
   408 unique primary-tumor patients).
3. Tertile-bins each of the four scores within the n=408 cohort using
   `pd.qcut` on rank. The headline `Subtype` column duplicates
   `Subtype_Decipher` (the curatedPCaData published Decipher score is
   the primary transcriptomic comparator). The AR-score tertiles are
   labelled `AR-low` / `AR-intermediate` / `AR-high` -- AR signalling
   activity is a biological-function score, not a metastasis-risk
   score, and "Low" should not implicitly mean "low risk."
4. Renames the raw score columns with an explicit `_curatedpcadata`
   suffix to flag provenance after the merge.
5. Left-joins the score table onto the clinical table on `Patient_ID`
   (408 of 500 PRAD patients carry curatedPCaData scores; the other 92
   have no primary-tumor RNA-seq aliquot in the curatedPCaData
   processing pipeline).

Run:

```bash
cd data/prognostic/prad
python build_prad_metadata.py     # requires `pyreadr`
```

Override paths via the `SCORES_RDS`, `CLINICAL_CSV`, and `OUTPUT_CSV`
environment variables.

## Output

`prad_clinical_metadata.csv` -- one row per PRAD patient (n=500),
columns:

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode. |
| `Subtype` | Within-cohort Decipher tertile (Low / Intermediate / High); same as `Subtype_Decipher`. |
| `Subtype_Decipher`, `Subtype_Oncotype`, `Subtype_Prolaris`, `Subtype_AR` | Within-cohort tertiles of each published curatedPCaData score. |
| `decipher_curatedpcadata`, `oncotype_curatedpcadata`, `prolaris_curatedpcadata`, `ar_score_curatedpcadata` | Continuous published scores. |
| Liu 2018 endpoints | `PFI`, `PFI.time`, `PFI.1`, `PFI.time.1`, `PFI.2`, `PFI.time.2`, `PFS`, `PFS.time`, `DSS_cr`, `DSS.time.cr`, `DFI.cr`, `DFI.time.cr`, `PFI.cr`, `PFI.time.cr`, `PFI.1.cr`, `PFI.time.1.cr`, `PFI.2.cr`, `PFI.time.2.cr`. |
| `curatedpcadata_resource` | Provenance string `EH8024 / tcga_scores_20230215.Rds`. |

## Caveats

- All four curatedPCaData scores are signature reconstructions on
  RNA-seq, not the proprietary clinical scores from the test vendors.
  Per-patient agreement with the proprietary scores has been validated
  in the curatedPCaData release paper but is not perfect.
- Tertile thresholds are derived within the TCGA-PRAD cohort, not
  anchored to published clinical thresholds. The published Decipher
  thresholds (Low <0.45 / Intermediate 0.45-0.6 / High >0.6 on the
  proprietary 0-1 score) **do not transfer** to the curatedPCaData
  score's distribution.
- TCGA-PRAD is enriched for early-stage, surgically-managed prostate
  cancer; results may not generalise to higher-risk cohorts (e.g.,
  post-RT salvage settings where Decipher is most clinically
  actionable).
