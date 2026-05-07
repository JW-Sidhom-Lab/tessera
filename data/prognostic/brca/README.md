# BRCA clinical metadata (PAM50 + research-reconstructed OncotypeDX / MammaPrint)

Per-patient PAM50 intrinsic subtype, AJCC pathologic stage, TCGA
Pan-Cancer Atlas survival endpoints, and research-reconstructed
OncotypeDX Recurrence Score and MammaPrint risk score for the TCGA-BRCA
cohort. Used by Figure 5 c-j of the manuscript: UMAP overlay of the
TESSERA breast-cancer manifold + joint Cox of the TESSERA risk score
against the OncotypeDX comparator.

## Inputs

| File | Source | Citation |
|---|---|---|
| `brca_subtype.tsv` | TCGA-BRCA 2012 supplementary | TCGA Network. Comprehensive molecular portraits of human breast tumours. *Nature* **490**, 61-70 (2012). |
| `DLRS_tcga_brca_complete.csv` | DLRS GitHub release | Howard, F. M. *et al.* Multimodal prediction of breast cancer recurrence assays and risk of recurrence. *npj Breast Cancer* (2023). DLRS: github.com/fmhoward/DLRS. |
| `../../TCGA_PanCan/clinical.csv` | TCGA Pan-Cancer Atlas curated clinical resource (Liu 2018-style) | Liu, J. *et al.* *Cell* **173**, 400-416.e11 (2018). |

`brca_subtype.tsv` provides the **PAM50 intrinsic subtype** call per
TCGA-BRCA tumor (`PAM50Call_RNAseq` column); patients without a PAM50
call are dropped. `Patient_ID` is the first 12 characters of the sample
barcode.

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

`build_brca_metadata.py` consolidates what was previously a two-step
pipeline (initial PAM50+ncit build, then a DLRS update) into one
self-contained script:

1. Loads `brca_subtype.tsv`, drops rows without a PAM50 call,
   deduplicates to one row per patient.
2. Loads `clinical.csv`, restricts to `type=='BRCA'`, remaps the
   ambiguous-cause event code `2` to `0` (censored) per the Liu 2018
   convention, casts events to `Int64`, and renames the three
   `_cr`-suffixed pairs to legacy column names (`DSS`/`DSS.time`,
   `DFI`/`DFI.time`, `PFI`/`PFI.time`).
3. Loads `DLRS_tcga_brca_complete.csv`, deduplicates to one row per
   patient (DLRS has multiple slide-level rows per patient with
   identical molecular RS), tertile-bins the continuous Oncotype RS
   into Low / Intermediate / High using within-cohort `pd.qcut` ranks.
4. Inner-joins PAM50 + survival on `Patient_ID` (n=842), then
   left-joins DLRS scores (n=817 of 842 carry an Oncotype RS).

Run:

```bash
cd data/prognostic/brca
python build_brca_metadata.py
```

Override paths via the `PAM50_TSV`, `DLRS_CSV`, `NCIT_CSV`, and
`OUTPUT_CSV` environment variables.

## Output

`brca_clinical_metadata.csv` -- one row per BRCA patient, columns:

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode. |
| `Subtype` | PAM50 intrinsic subtype (`LumA`, `LumB`, `Basal`, `Her2`, `Normal`). |
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
- OS and `ajcc_pathologic_tumor_stage` are intentionally not emitted:
  `clinical.csv` only ships Liu 2018 curated DSS / DFI / PFI columns
  and does not carry stage. None of the manuscript Figure 4 or
  Figure 5 BRCA analyses use OS or stage.
