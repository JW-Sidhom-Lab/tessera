# BRCA clinical metadata (research-reconstructed OncotypeDX / MammaPrint)

Per-patient TCGA Pan-Cancer Atlas survival endpoints + research-
reconstructed OncotypeDX Recurrence Score and MammaPrint risk score
for the TCGA-BRCA cohort. Used by Figure 5 c-j of the manuscript:
UMAP overlay of the TESSERA breast-cancer manifold + joint Cox of the
TESSERA risk score against the OncotypeDX comparator.

## Inputs

| File | Source | Citation |
|---|---|---|
| `DLRS_tcga_brca_complete.csv` | DLRS GitHub release | Howard, F. M. *et al.* Multimodal prediction of breast cancer recurrence assays and risk of recurrence. *npj Breast Cancer* (2023). DLRS: github.com/fmhoward/DLRS. |
| `../../TCGA_PanCan/clinical.csv` | TCGA Pan-Cancer Atlas curated clinical resource (Liu 2018-style) | Liu, J. *et al.* *Cell* **173**, 400-416.e11 (2018). |

`DLRS_tcga_brca_complete.csv` is the per-patient release from the Deep
Learning for Recurrence Score (DLRS) repository accompanying Howard et
al. 2023. DLRS provides:

- **OncotypeDX Recurrence Score** (`odx_train` column): the Paik 2004
  21-gene formula applied to TCGA-BRCA RNA-seq. This is a research
  reconstruction, **not** the proprietary clinical score from Genomic
  Health.
- **OncotypeDX 85th-percentile binary** (`odx85`): top 15% = `H`,
  bottom 85% = `L`.
- **MammaPrint score** (`mp_train`) and three binarizations
  (`mphr` high-risk, `mpulr` ultra-low-risk, `mpuhr` ultra-high-risk).

`clinical.csv` is the Liu 2018 curated TCGA Pan-Cancer Atlas clinical
resource, gitignored as a raw upstream input under
[`data/TCGA_PanCan/`](../../TCGA_PanCan/README.md). For BRCA we pull
the three curated endpoint pairs (`DSS_cr`/`DSS.time.cr`,
`DFI.cr`/`DFI.time.cr`, `PFI.cr`/`PFI.time.cr`).

## Processing

`build_brca_metadata.py`:

1. Loads `clinical.csv`, restricts to `type=='BRCA'`, remaps the
   ambiguous-cause event code `2` to `0` (censored) per the Liu 2018
   convention, casts events to `Int64`, and renames the three
   `_cr`-suffixed pairs to legacy column names (`DSS`/`DSS.time`,
   `DFI`/`DFI.time`, `PFI`/`PFI.time`).
2. Loads `DLRS_tcga_brca_complete.csv`, deduplicates to one row per
   patient (DLRS has multiple slide-level rows per patient with
   identical molecular RS), tertile-bins the continuous Oncotype RS
   into Low / Intermediate / High using within-cohort `pd.qcut` ranks.
3. Left-joins DLRS onto the survival table (n=1,097 BRCA patients;
   1,039 of them carry DLRS Oncotype RS).

Run:

```bash
cd data/prognostic/brca
python build_brca_metadata.py
```

Override paths via the `DLRS_CSV`, `CLINICAL_CSV`, and `OUTPUT_CSV`
environment variables.

## Output

`brca_clinical_metadata.csv` -- one row per BRCA patient (n=1,097),
columns:

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode. |
| `DSS`, `DSS.time` | Liu 2018 curated disease-specific survival (`DSS_cr`/`DSS.time.cr` with `2 -> 0` remap). |
| `DFI`, `DFI.time` | Liu 2018 curated disease-free interval. |
| `PFI`, `PFI.time` | Liu 2018 curated progression-free interval. |
| `oncotype_rs_dlrs` | DLRS-reconstructed continuous OncotypeDX Recurrence Score (Paik 2004 formula on TCGA-BRCA RNA-seq). |
| `oncotype_high_85th` | DLRS-binarized OncotypeDX (top 15% = `H`). |
| `mammaprint_score_dlrs`, `mammaprint_high`, `mammaprint_ultra_low`, `mammaprint_ultra_high` | DLRS-reconstructed MammaPrint score and binary flags. |
| `Subtype_Oncotype` | Within-cohort tertile of `oncotype_rs_dlrs` (Low / Intermediate / High). |

## Caveats

- `oncotype_rs_dlrs` and the MammaPrint score are research
  reconstructions from RNA-seq, not the proprietary clinical scores.
  Per-patient agreement with the proprietary scores has been validated
  in the DLRS release paper but is not perfect.
- Tertile thresholds for `Subtype_Oncotype` are within-cohort and do
  not correspond to the clinical Recurrence Score thresholds (RS<18 /
  18-30 / >=31).
- PAM50 intrinsic subtype, OS, and `ajcc_pathologic_tumor_stage` are
  intentionally not emitted: none are used by Figure 5 BRCA. An
  earlier version of this build inner-joined on PAM50, which silently
  dropped 222 BRCA patients with DLRS Oncotype RS but no PAM50 call;
  removing that filter recovers them.
