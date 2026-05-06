# MSK-CHORD model-ready SNV and CNA tables

Builds the SNV and CNA tables that feed the MSK-CHORD CNA cross-platform
validation (Figure 2 d) and the clinical predictive-biomarker analyses
(Figure 6 b-i).

## Pipeline

```
data/msk_chord_2024/msk_chord_2024.csv ─→ create_snv.py ─→ snv.csv

data/msk_chord_2024/data_cna_hg19.seg ──┐
data/msk_chord_2024/                    │
data_gene_panel_matrix.txt ─────────────┼─→ create_cna.py ─→ cna.csv
data/genie_18_0/                        │                    cna_panel_filtered.csv
genomic_information.txt ────────────────┘
```

## Running

```bash
# SNV: filter and subsample (~1 minute)
python create_snv.py

# CNA: write raw segments and panel-trimmed segments (~5 minutes)
python create_cna.py
```

The two scripts are independent and can run in parallel.

## Outputs

| File | Description |
|---|---|
| `snv.csv` | Per-variant rows for clinical / cross-platform analyses (~115 MB, gitignored). |
| `cna.csv` | Raw CNA segments for solid-tumor samples in IMPACT341/410/468/505 (excludes IMPACT-HEME-400). Columns: `Tumor_Sample_Barcode, Chromosome, Start, End, Segment_Mean`. ~59 MB, gitignored. |
| `cna_panel_filtered.csv` | Same segments trimmed to each sample's panel-covered gene regions. Adds `Hugo_Symbol` and `Panel` columns. ~694 MB, gitignored. |

## Key parameters

- **SNV filter:** `Variant_Type == 'SNP'`. Subsampled to up to 100 variants
  per sample, preserving variants seen in >= 5 samples cohort-wide.
- **CNA panel set:** IMPACT341, IMPACT410, IMPACT468, IMPACT505 (the four
  solid-tumor MSK-IMPACT panels). IMPACT-HEME-400 (hematologic) excluded.
- **Panel filtering:** segments are trimmed to the per-gene min/max-bounded
  region for each sample's panel, using gene coordinates aggregated from
  the GENIE 18.0 `genomic_information.txt` file. The four IMPACT panels are
  stable across GENIE releases, so the choice of release version does not
  affect outputs.
