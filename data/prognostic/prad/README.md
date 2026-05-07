# PRAD clinical metadata (curatedPCaData transcriptomic comparators)

Per-patient Decipher / OncotypeDX / Prolaris / AR-signaling scores
from the published curatedPCaData TCGA release, plus TCGA-PRAD curated
clinical / outcome columns. Used by Figure 5 k-r (UMAP overlay + joint
Cox of the TESSERA risk score against the Decipher comparator).

## Inputs

| File | Source |
|---|---|
| `curatedPCaData_tcga_scores_20230215.Rds` | [Laajala et al. 2023, *Scientific Data* **10**, 430](https://doi.org/10.1038/s41597-023-02335-4). Four transcriptomic signature scores computed under a single uniform RNA-seq pipeline (table below). Hosted as Bioconductor ExperimentHub resource `EH8024` ([package page](https://bioconductor.org/packages/curatedPCaData/)); the file in this directory was downloaded from the OSN bucket at [`mghp.osn.xsede.org/.../tcga_scores_20230215.Rds`](https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/curatedPCaData/tcga_scores_20230215.Rds). |
| `../../TCGA_PanCan/clinical.csv` | [Liu et al. 2018, *Cell* **173**, 400-416.e11](https://doi.org/10.1016/j.cell.2018.02.052). Curated PRAD endpoints. `PFI.1` is the Liu 2018 recommended primary endpoint for PRAD (DSS is underpowered, ~5 events). |

| Score column | Signature |
|---|---|
| `decipher` | 22-gene metastasis classifier (Erho 2013, Karnes 2013). |
| `oncotype` | OncotypeDX GPS, 17-gene (Knezevic 2013). |
| `prolaris` | Cuzick CCP cell-cycle-progression (Cuzick 2011). |
| `ar_score` | 20-gene AR-signaling activity (Hieronymus 2006). |

## Processing

`build_prad_metadata.py`:

1. Load `clinical.csv`, restrict to `type=='PRAD'`.
2. Load `curatedPCaData_tcga_scores_20230215.Rds` via `pyreadr`, keep
   primary-tumor aliquots (sample-type `01`), aggregate replicates per
   patient by mean.
3. Tertile-bin each score within the cohort. AR-score tertiles are
   labelled `AR-low` / `AR-intermediate` / `AR-high` because AR
   signalling is a biological-activity score, not a metastasis-risk
   score.
4. Left-join the score table onto the clinical table on `Patient_ID`.

```bash
cd data/prognostic/prad
python build_prad_metadata.py     # requires `pyreadr`
```

Path overrides: `SCORES_RDS`, `CLINICAL_CSV`, `OUTPUT_CSV`.

## Output

`prad_clinical_metadata.csv`, one row per PRAD patient:

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode. |
| `Subtype` | Headline categorical comparator (same as `Subtype_Decipher`). |
| `Subtype_Decipher`, `Subtype_Oncotype`, `Subtype_Prolaris`, `Subtype_AR` | Within-cohort tertiles of each transcriptomic score. |
| `decipher_curatedpcadata`, `oncotype_curatedpcadata`, `prolaris_curatedpcadata`, `ar_score_curatedpcadata` | Continuous published scores. |
| Liu 2018 endpoints | `PFI`, `PFI.time`, `PFI.1`, `PFI.time.1`, `PFI.2`, `PFI.time.2`, `PFS`, `PFS.time`, `DSS_cr`, `DSS.time.cr`, `DFI.cr`, `DFI.time.cr`, `PFI.cr`, `PFI.time.cr`, `PFI.1.cr`, `PFI.time.1.cr`, `PFI.2.cr`, `PFI.time.2.cr`. |
| `curatedpcadata_resource` | Provenance string. |

## Caveats

- All four curatedPCaData scores are RNA-seq reconstructions, not the
  proprietary clinical scores. Agreement with the proprietary scores
  is validated in the curatedPCaData release paper but is not perfect.
- Tertile thresholds are within-cohort and do not transfer to
  published vendor thresholds (e.g., the proprietary Decipher
  Low&nbsp;&lt;&nbsp;0.45 / Intermediate&nbsp;0.45-0.6 /
  High&nbsp;&gt;&nbsp;0.6 cuts on the 0-1 commercial score).
- TCGA-PRAD is enriched for early-stage, surgically-managed prostate
  cancer; results may not generalise to higher-risk cohorts.
