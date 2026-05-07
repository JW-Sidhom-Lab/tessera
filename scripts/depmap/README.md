# DepMap CRC cell-line validation (Figure 6 n)

Applies the frozen MSK-CHORD-trained CRC predictive biomarker (tau_hat,
beta_eff x + const_eff) to DepMap 24Q2 colorectal cell lines and tests
its Spearman correlation with the FOLFOX-over-FOLFIRI z-scored
sensitivity preference computed from CTRPv2 oxaliplatin and SN-38
dose-response AUCs. Backs Fig. 6 n + Methods §"DepMap cell-line
validation".

## Pipeline

```
../data/depmap/snv_panel.csv + cna_panel.csv          (panel-restricted DepMap inputs from scripts/data/depmap/prepare_depmap_data.py)
../tcga_pancan_snv_cna/models/...InfoNCE_per_sample_loss_noLOH/best_model.keras
                       │
                       ▼ get_latent_features.py                                    [RunPod]
       depmap_latent_features_panel.pkl
                       │
                       │ + ../predictive_bm/cache/msk_chord_scalers.pkl
                       │ + ../predictive_bm/attribution_analysis/crc_signatures/attribution_matrix.pkl
                       ▼ apply_tau.py                                              [local]
       results/tau/depmap_tau.tsv
                       │
                       │ joined with ../data/depmap/metadata.csv (lineage, drug AUCs)
                       ▼ analyze_depmap.py                                         [local]
       results/analysis/preference_per_cell_line.tsv
       results/analysis/preference_test.tsv
                       │
                       ▼ manuscript/build/figure6_signature_panels.py             (renders Fig 6 n)
```

## Files

| Script | What it does |
|---|---|
| [`get_latent_features.py`](get_latent_features.py) | Frozen TESSERA inference on the MSK-IMPACT505 panel-restricted DepMap SNV / CNA tables; emits `depmap_latent_features_panel.pkl`. |
| [`apply_tau.py`](apply_tau.py) | Loads the panel features pickle + the MSK-CHORD RobustScalers + `beta_eff`, `const_eff`, `tau0`. Builds the per-cell-line genomic fingerprint and computes `tau_hat = beta_eff @ x + const_eff`. Joins with `metadata.csv` (lineage, drug AUC) and writes `results/tau/depmap_tau.tsv`. |
| [`analyze_depmap.py`](analyze_depmap.py) | For each CRC cell line with both oxaliplatin and SN-38 AUC measured, computes the FOLFOX-over-FOLFIRI z-scored sensitivity preference, and runs a one-sided Spearman test of `tau_hat` vs preference (predicted positive direction). Writes `preference_per_cell_line.tsv` + `preference_test.tsv`. |

## Why panel-restricted inputs

The TESSERA SNV and CNA encoders use cross-token attention. Inference on
full WES vs MSK-IMPACT505-filtered inputs produces different per-token
embeddings even for the same variant, because each token attends over a
different bag. The MSK-CHORD-fitted `tau_hat` map (`beta_eff`) was learned
on MSK-IMPACT505 panel coverage, so for a cross-cohort transfer, DepMap
inference must run on the same panel-restricted bag. The panel-restricted
SNV / CNA tables in `../data/depmap/{snv_panel,cna_panel}.csv` are produced
by `scripts/data/depmap/prepare_depmap_data.py` for this purpose.

## Running it

```bash
cd scripts/depmap

# 1. Frozen-model inference on DepMap (one-time, ~451 MB output, RunPod)
python3 get_latent_features.py

# 2. Compute tau_hat per cell line (local, seconds)
python3 apply_tau.py

# 3. Compute Fig 6 n preference correlation (local, seconds)
python3 analyze_depmap.py
```
