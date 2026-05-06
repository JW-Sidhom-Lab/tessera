# GENIE model-ready SNV table

Builds the per-sample SNV subset of the GENIE 18.0 release used for the
cross-platform SNV-transfer evaluation reported in Figure 1 f-g of the
manuscript.

## Pipeline

```
data/genie_18_0/GENIE.csv  ────→ create_snv.py ──→ snv.csv
```

## Running

```bash
python create_snv.py
```

Defaults assume the upstream input sits at
`../../../data/genie_18_0/GENIE.csv`. Override with `--variants` and
`--output` if your layout differs.

## Output

| File | Description |
|---|---|
| `snv.csv` | Per-variant rows for SNV-transfer evaluation. All upstream columns from `GENIE.csv` plus `mut_id` (locus key). Gitignored (~1 GB). |

## Key parameters

- **Variant filter:** `Variant_Type == 'SNP'` (single-nucleotide variants).
  Variants without a defined `vaf` are also dropped.
- **Subsampling:** up to 100 variants per sample. Variants seen in >= 5
  samples cohort-wide are preserved as recurrent alterations; the rest of
  the budget is filled by uniform random sampling within the sample with
  `seed = 42`.

## Note on CNA

GENIE CNA segments are not processed in this directory. CNA cross-platform
validation in the manuscript uses MSK-CHORD (`scripts/data/msk_chord/`), so
the original `methods/data/genie/create_cna.py` and `subdivide_cna.py` were
not migrated.
