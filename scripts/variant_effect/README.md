# Variant-effect prediction (ClinVar pathogenicity)

Reproduces Figure 1 h-o of the manuscript: ClinVar pathogenic-vs-benign
variant classification using TESSERA per-variant latent features. The 7
TESSERA SNV models from
[`scripts/tcga_pancan_snv/`](../tcga_pancan_snv/README.md) are evaluated
under four splits, by combining two **dedup strategies**
(variant-level vs gene-level) with two **subsetting choices**
(all variants vs only variants whose masked ref/alt were correctly
reconstructed during pretraining).

| Split strategy | Subset | Manuscript figure |
|---|---|---|
| variant | all | Fig. 1 h-i |
| variant | correctly reconstructed | Fig. 1 j-k |
| gene | all | Fig. 1 l-m |
| gene | correctly reconstructed | Fig. 1 n-o |

The variant strategy tests generalization to new mutations in known
genes; the gene strategy tests generalization to completely unseen genes.

## Pipeline

```
scripts/data/tcga/{train,valid}_data_snv.csv     (upstream, model-ready CSVs)
                          │
                          ▼
              create_variant_dedup_split.py
                          │
                          ▼
        data/{train,valid}_data_snv_dedup{,_gene}.csv
                          │
                          ▼
              create_index_mapping.py
                          │
                          ▼
              data/index_mapping{,_gene}.pkl
                          │
                          ▼
        generate_clinvar_predictions.py    (uses ClinVar + TESSERA features)
                          │
                          ▼
        clinvar_preds_dedup_<strategy>/<config>_{train,valid}_predictions.csv
                          │
                          ▼
        analyze_clinvar_predictions.py     (Figure 1 h-o)
```

## Running

```bash
# Stage 1: build both dedup splits and the index mappings
python create_variant_dedup_split.py
python create_index_mapping.py

# (optional) sanity-check the splits
STRATEGY=variant python validate_split.py
STRATEGY=gene    python validate_split.py

# Stage 2: per-model ClinVar predictions, both strategies
STRATEGY=variant python generate_clinvar_predictions.py
STRATEGY=gene    python generate_clinvar_predictions.py

# Stage 3: figures (Figure 1 h-o)
STRATEGY=variant python analyze_clinvar_predictions.py
STRATEGY=gene    python analyze_clinvar_predictions.py
```

`generate_clinvar_predictions.py` reads:

- ClinVar labels from [`data/clinvar/clinvar.pkl`](../../data/clinvar/README.md)
- per-variant latent features from `scripts/tcga_pancan_snv/var_features/`
  (produced by `get_variant_features.py`)
- per-variant masked-token predictions from `scripts/tcga_pancan_snv/var_loss/`
  (produced by `get_variant_loss_acc.py`), used to build the
  "correctly reconstructed" subset

so the upstream SNV pipeline must have run before this analysis.

### Recognised env vars

| Variable | Used in | Notes |
|---|---|---|
| `STRATEGY` | all scripts | `variant` / `gene` (or `both` for the create_*.py drivers) |
| `TRAIN_DATA`, `VALID_DATA` | create_* scripts | upstream patient-level paths |
| `OUTPUT_DIR` | create_*, analyze | dedup-split or figures output dir |
| `PREDICTION_DIR` | analyze_clinvar_predictions.py | input directory of `*_predictions.csv` |
| `N_BOOTSTRAP`, `CONFIDENCE_LEVEL` | analyze_clinvar_predictions.py | bootstrap settings |
| `REQUIRE_CORRECT_PREDICTIONS` | analyze_clinvar_predictions.py | `0` to skip the correctly-reconstructed subsetting |
| `BASELINE_MODEL` | analyze_clinvar_predictions.py | comparison baseline (default `baseline`) |
| `TRAIN_RATIO`, `RANDOM_SEED` | create_variant_dedup_split.py | split ratio + seed (default 0.75 / 42) |

## Outputs

| Path | Description |
|---|---|
| `data/{train,valid}_data_snv_dedup{,_gene}.csv` | Dedup-split SNV tables |
| `data/index_mapping{,_gene}.pkl` | Dedup-row to original-feature-index map |
| `data/split_report{,_gene}.txt` | Split summary (sizes, overlap, integrity) |
| `clinvar_preds_dedup_<strategy>/<config>_{train,valid}_predictions.csv` | Per-model logistic-regression predictions on the ClinVar subset |
| `plots/clinvar_dedup_<strategy>/` | ROC + PR figures |

The pipeline is deterministic given a fixed `RANDOM_SEED`; re-running on
the same upstream inputs reproduces all figures.
