# TCGA CNA pretraining and analysis

Trains the three TESSERA CNA-only models reported in Figure 2 of the
manuscript (segment-mean reconstruction + optional LOH classification,
varying inter-segment self-attention depth: 0, 1, 2 blocks) and extracts
the per-segment predictions, latent features, and reconstruction figures
that downstream analyses consume.

## Pipeline

```
data/TCGA_PanCan/TCGA_CNA.csv                 (Stage 1, scripts/data/tcga/)
                       │
                       ▼
        scripts/data/tcga/train_data_cna.csv,
        scripts/data/tcga/valid_data_cna.csv  (Stage 2, model-ready CSVs)
                       │
                       ▼
                  fit_model.py    (3 models trained)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
get_cna_features.py            get_cna_loss_metrics.py
       │                               │
       ▼                               ▼
cna_features/                  cna_loss/
                                       │
                                       ▼
                       plot_cna_reconstruction.py  (Figure 2 a-c)
```

## Configurations

Three architectures, switched via `CNA_ATTN_BLOCKS=N`:

| `cna_attention_blocks` | Model name | Inter-segment attention |
|---|---|---|
| 0 | `TCGA_PanCan_CNA_attn_0` | none |
| 1 | `TCGA_PanCan_CNA_attn_1` | one self-attention block |
| 2 | `TCGA_PanCan_CNA_attn_2` | two self-attention blocks |

Each model has a segment-mean regression head plus an LOH classification
head (when `predict_cna_loh=True` in `model_config.py`). The manuscript
also reports a NoLOH variant used for the MSK-CHORD cross-platform
validation (Figure 2 d) where allele-specific copy-number is unavailable;
flip `predict_cna_loh` to reproduce it.

## Running

```bash
# Train all 3 models reported in Figure 2
./run_all_configs.sh

# After training, extract metrics and features
./run_get_loss.sh
./run_get_features.sh

# Generate Figure 2 a-c
python plot_cna_reconstruction.py

# Or run the full pipeline (train -> loss -> features) in one go
./run_pipeline.sh
```

Or run individual configs:

```bash
python fit_model.py                                     # attn_0 (default)
CNA_ATTN_BLOCKS=1 python fit_model.py                   # attn_1
CNA_ATTN_BLOCKS=2 python fit_model.py                   # attn_2

CNA_ATTN_BLOCKS=2 python get_cna_loss_metrics.py
CNA_ATTN_BLOCKS=2 python get_cna_features.py
```

The shell-script entry points accept custom subsets:

```bash
./run_all_configs.sh 2          # train one model
./run_get_features.sh 1 2       # extract features for two models
```

### Recognised env vars

| Variable | Used in | Notes |
|---|---|---|
| `CNA_ATTN_BLOCKS` | all 3 model scripts | `0`, `1`, or `2` |
| `TRAIN_DATA`, `VALID_DATA` | all model scripts | override default `../data/tcga/...` paths |
| `OUTPUT_DIR` | `get_cna_*` and plot script | override default `cna_features` / `cna_loss` / `plots/cna_analysis` |
| `LOSS_DIR` | `plot_cna_reconstruction.py` | input directory of `*_cna_metrics.pkl` files |
| `ATTN_BLOCKS_TO_INCLUDE` | `plot_cna_reconstruction.py` | comma-separated list, default `0,1,2` |
| `MATPLOTLIB_BACKEND` | plot script | e.g. `Agg` for headless |

## Outputs

| Directory | Contents | Tracked? |
|---|---|---|
| `models/<model_name>/` | Trained TESSERA CNA model artefacts | no |
| `cna_features/<model_name>_cna_features.pkl` | Per-segment latent features (train + valid) | no |
| `cna_loss/<model_name>_cna_metrics.pkl` | Predicted vs actual segment-mean + LOH; summary metrics (MSE/MAE/R^2/correlation; LOH accuracy/AUC/precision/recall/F1) | no |
| `plots/cna_analysis/` | Generated Figure 2 panels (PNG) | no |

All large output directories are gitignored. Re-running the pipeline on
the same Stage-2 CSVs reproduces them deterministically (seeds pinned).

## Compute requirements

Models are trained on an NVIDIA RTX 6000 Ada.
