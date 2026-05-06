# TCGA model-ready train/valid splits

Builds the patient-level train/validation split for TCGA pretraining, then
subsamples variants and CNA segments per sample into the four CSVs the
TESSERA SNV and CNA encoders consume.

## Pipeline

```
data/TCGA_PanCan/clinical.csv ──→ create_train_test_patients.py
                                         │
                                         ▼
                          train_test_patients_reference.csv
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
data/TCGA_PanCan/                                       
TCGA.csv      ────→ create_training_snv.py ──→ train_data_snv.csv
                                                valid_data_snv.csv

data/TCGA_PanCan/                                       
TCGA_CNA.csv  ────→ create_training_cna.py ──→ train_data_cna.csv
                                                valid_data_cna.csv
```

## Running

Stage 1 must run first; the two Stage-2 scripts are independent and can run
in parallel.

```bash
# 75/25 patient split stratified by tumour type
python create_train_test_patients.py

# SNV train/valid split + per-sample subsampling
python create_training_snv.py

# CNA train/valid split + per-sample subsampling
python create_training_cna.py
```

Defaults assume the upstream raw inputs sit at `../../../data/TCGA_PanCan/`.
Override with `--variants`, `--cna`, `--clinical`, `--reference`, or
`--output-dir` if your layout differs.

## Outputs

| File | Description |
|---|---|
| `train_test_patients_reference.csv` | Per-patient split assignment (`bcr_patient_barcode`, `type`, `split`). Committed to lock the exact split used in the manuscript. |
| `train_data_snv.csv` / `valid_data_snv.csv` | Per-variant rows for SNV pretraining. Columns include sample/patient barcodes, locus, alleles, VAF, variant class, gene, and tumour type. |
| `train_data_cna.csv` / `valid_data_cna.csv` | Per-segment rows for CNA pretraining. Columns: `Tumor_Sample_Barcode, Chromosome, Start, End, Segment_Mean, LOH`. |

The four train/valid CSVs are gitignored (30-150 MB each) and regenerated
deterministically from the upstream `TCGA.csv` / `TCGA_CNA.csv` plus the
committed reference.

## Key parameters

Defaults match the manuscript. Override via CLI flags to explore alternatives.

- **Tumour-type filter** (`create_train_test_patients.py`): 31 solid tumour
  types; hematologic cancers are excluded. See `SOLID_TUMOR_TYPES`.
- **Train/test split**: 75/25 stratified by tumour type, `seed = 42`.
- **SNV subsampling**: up to 1,000 variants per sample; variants recurrent
  in >= 5 samples are preserved, then a uniform random sample fills the
  remainder.
- **CNA subsampling**: up to 1,000 segments per sample, ranked by
  `|Segment_Mean| + 0.5 * LOH`.
- **Patient barcode**: first 12 characters of `Tumor_Sample_Barcode`
  (TCGA's `TCGA-XX-NNNN` patient prefix).
