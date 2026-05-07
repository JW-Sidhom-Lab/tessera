# BRCA clinical metadata (research-reconstructed OncotypeDX / MammaPrint)

Per-patient TCGA Pan-Cancer Atlas survival endpoints plus
research-reconstructed OncotypeDX Recurrence Score and MammaPrint risk
score for the TCGA-BRCA cohort. Used by Figure 5 c-j (UMAP overlay +
joint Cox of the TESSERA risk score against the OncotypeDX
comparator).

## Inputs

| File | Source |
|---|---|
| `DLRS_tcga_brca_complete.csv` | Howard et al. 2023, *npj Breast Cancer*. DLRS GitHub: `github.com/fmhoward/DLRS`. RNA-seq reconstructions of OncotypeDX RS (Paik 2004 21-gene formula) and MammaPrint. |
| `../../TCGA_PanCan/clinical.csv` | Liu et al. 2018, *Cell* **173**, 400-416.e11. Curated DSS / DFI / PFI endpoints. |

## Processing

`build_brca_metadata.py`:

1. Load `clinical.csv`, restrict to `type=='BRCA'`, remap each curated
   event code `2 -> 0` (Liu 2018 ambiguous-cause -> censored), rename
   the `_cr`-suffixed pairs to legacy names.
2. Load `DLRS_tcga_brca_complete.csv`, deduplicate to one row per
   patient (DLRS is per-slide), tertile-bin the continuous OncotypeDX
   RS within the DLRS cohort into Low / Intermediate / High.
3. Left-join DLRS scores onto the survival table.

```bash
cd data/prognostic/brca
python build_brca_metadata.py
```

Path overrides: `DLRS_CSV`, `CLINICAL_CSV`, `OUTPUT_CSV`.

## Output

`brca_clinical_metadata.csv`, one row per BRCA patient:

| Column | Description |
|---|---|
| `Patient_ID` | TCGA patient barcode. |
| `DSS`, `DSS.time` | Disease-specific survival event (binary) and time (Liu 2018 curated). |
| `DFI`, `DFI.time` | Disease-free interval. |
| `PFI`, `PFI.time` | Progression-free interval. |
| `oncotype_rs_dlrs` | DLRS-reconstructed continuous OncotypeDX RS. |
| `oncotype_high_85th` | DLRS-binarized OncotypeDX (top 15% = `H`). |
| `mammaprint_score_dlrs`, `mammaprint_high`, `mammaprint_ultra_low`, `mammaprint_ultra_high` | DLRS-reconstructed MammaPrint score + binary flags. |
| `Subtype_Oncotype` | Within-cohort tertile of `oncotype_rs_dlrs` (Low / Intermediate / High). |

## Caveats

- The OncotypeDX and MammaPrint columns are RNA-seq-based research
  reconstructions, not the proprietary clinical scores. Agreement with
  the proprietary scores is validated in the DLRS release paper but is
  not perfect.
- `Subtype_Oncotype` tertile thresholds are within-cohort and do not
  correspond to the clinical Recurrence Score thresholds
  (RS&lt;18 / 18-30 / &ge;31).
