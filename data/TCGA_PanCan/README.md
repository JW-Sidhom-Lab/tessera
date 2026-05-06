# TCGA Pan-Cancer Atlas data preparation

This directory holds the data-preparation pipeline for the TCGA Pan-Cancer Atlas
cohort, the pretraining corpus for TESSERA. Two scripts produce the somatic-
variant table (`TCGA.csv`) and the copy-number segment table (`TCGA_CNA.csv`)
consumed by the model training code.

## Required raw inputs

Download from the TCGA Pan-Cancer Atlas publication page on GDC:
**https://gdc.cancer.gov/about-data/publications/pancanatlas**

No login required.

| File | Size | Source paper |
|---|---|---|
| `mc3.v0.2.8.PUBLIC.maf` | 3.5 GB | Ellrott et al., *Cell Systems* 2018 (MC3) |
| `TCGA_mastercalls.abs_segtabs.fixed.txt` | 241 MB | Carter et al., *Nat. Biotech.* 2012 (ABSOLUTE) |
| `TCGA_mastercalls.abs_tables_JSedit.fixed.txt` | 881 KB | Same (ABSOLUTE purity/ploidy table) |
| `TCGA-CDR-SupplementalTableS1.xlsx` | 2.8 MB | Liu et al., *Cell* 2018 (Pan-Cancer Clinical Data Resource) |

Place all four files in this directory.

## Optional pre-derived clinical CSV

`clinical.csv` is a CSV export of the xlsx's `ExtraEndpoints` sheet
(`bcr_patient_barcode`, `type`, plus PFI/PFS/DSS/DFI columns). The SNV pipeline
reads this CSV, so either:

- export the sheet yourself (any tool), or
- use the existing `clinical.csv` if you already have one alongside the xlsx.

The columns must include `bcr_patient_barcode` and `type`.

## Running the pipeline

```bash
# Stage 1: somatic variants (~5 minutes; produces TCGA.csv ~281 MB)
python create_training_data.py \
    --maf mc3.v0.2.8.PUBLIC.maf \
    --clinical clinical.csv \
    --output TCGA.csv

# Stage 2: copy-number segments (~2 minutes; produces TCGA_CNA.csv ~138 MB)
python create_training_data_cna.py \
    --segments TCGA_mastercalls.abs_segtabs.fixed.txt \
    --purity TCGA_mastercalls.abs_tables_JSedit.fixed.txt \
    --output TCGA_CNA.csv
```

The two stages are independent and may run in parallel.

## Outputs

| File | Columns | Used by |
|---|---|---|
| `TCGA.csv` | `Tumor_Sample_Barcode, Hugo_Symbol, Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2, vaf, VARIANT_CLASS, HGVSp_Short, bcr_patient_barcode, type` | downstream SNV train/valid split |
| `TCGA_CNA.csv` | `Sample, Chromosome, Start, End, Segment_Mean, LOH` | downstream CNA train/valid split |

`Segment_Mean` is the purity-adjusted log2 copy-ratio: observed CN is computed
as `purity * Modal_Total_CN + (1 - purity) * 2`, then expressed as
`log2(observed_CN / 2)`. This is the value the CNA encoder ingests as
`cna_segment_mean`.

## Memory notes

`create_training_data.py` loads the full 3.5 GB MC3 MAF into pandas memory; peak
usage is around 35-40 GB. `create_training_data_cna.py` is lighter (~2-4 GB).
