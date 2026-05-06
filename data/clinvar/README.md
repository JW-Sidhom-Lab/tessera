# ClinVar data preparation

This directory holds the data-preparation pipeline for ClinVar, the source of
variant-pathogenicity labels used in the manuscript's variant-effect prediction
task (Figure 1 h-o).

## Required raw input

Download the GRCh37 ClinVar VCF release dated **2024-12-30** (the version used
in the manuscript) from NCBI:

**https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/**

The file is named `clinvar_<YYYYMMDD>.vcf.gz`. Place it in this directory.
Newer releases are likely backwards-compatible (the same `INFO` fields are
parsed), but classifications drift over time, so locking to the manuscript
date avoids label churn.

| File | Size |
|---|---|
| `clinvar_20241230.vcf.gz` | 100 MB |

No login required. The script reads `.vcf.gz` directly; no manual decompression
step.

## Running the pipeline

```bash
python create_training_data.py \
    --vcf clinvar_20241230.vcf.gz \
    --output clinvar.pkl
```

Single stage, around 5 minutes, produces a 265 MB pickle.

## Output

| File | Format | Contents |
|---|---|---|
| `clinvar.pkl` | pickled pandas DataFrame | Standard VCF columns (`CHROM, POS, ID, REF, ALT, QUAL, FILTER`) plus three fields parsed out of `INFO`: `ALLELEID`, `CLNSIG` (pathogenicity classification, the prediction target), `GENEINFO` |

## Citation

Landrum et al. *ClinVar: improvements to accessing data.* Nucleic Acids
Research, 2020.
