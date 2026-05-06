# TCGA CNA NoLOH pretraining and analysis

NoLOH variant of the CNA pretraining pipeline. Trains TESSERA CNA models
with **only the segment-mean regression head** (no allele-specific LOH
classification), so the resulting model can be transferred to panel-
sequencing cohorts that don't carry LOH calls. This is the variant the
manuscript uses for the CNA cross-platform validation in Figure 2 d
(MSK-CHORD), Figure 6 b-i (clinical analyses), and Figure 6 n (DepMap
cell-line orthogonal validation).

The structure mirrors [`tcga_pancan_cna/`](../tcga_pancan_cna/README.md);
the difference is that LOH is dropped from both the input
(`cna_lohs=None`) and the output head (`predict_cna_loh=False`).

## Pipeline

```
scripts/data/tcga/train_data_cna.csv
scripts/data/tcga/valid_data_cna.csv     (model-ready CSVs from upstream)
                       │
                       ▼
                  fit_model.py    (3 NoLOH models trained)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
get_cna_features.py            get_cna_loss_metrics.py
       │                               │
       ▼                               ▼
cna_features/                  cna_loss/
                                       │
                                       ▼
                       plot_cna_reconstruction.py
```

## Configurations

Three architectures, switched via `CNA_ATTN_BLOCKS=N`:

| `cna_attention_blocks` | Model name | Inter-segment attention |
|---|---|---|
| 0 | `TCGA_PanCan_CNA_NoLOH_attn_0` | none |
| 1 | `TCGA_PanCan_CNA_NoLOH_attn_1` | one self-attention block |
| 2 | `TCGA_PanCan_CNA_NoLOH_attn_2` | two self-attention blocks (default; used for Fig. 2 d / 6 b-i / 6 n) |

## Running

```bash
# Train all 3 NoLOH models
./run_all_configs.sh

# After training, extract metrics + features
./run_get_loss.sh
./run_get_features.sh

# Plot segment-mean reconstruction (no LOH ROC since no LOH head)
python plot_cna_reconstruction.py

# Or run the full pipeline in one go
./run_pipeline.sh
```

Or run individual configs:

```bash
python fit_model.py                                     # attn_2 (default)
CNA_ATTN_BLOCKS=0 python fit_model.py
CNA_ATTN_BLOCKS=1 python fit_model.py

CNA_ATTN_BLOCKS=2 python get_cna_loss_metrics.py
CNA_ATTN_BLOCKS=2 python get_cna_features.py
```

### Recognised env vars

Same set as [`tcga_pancan_cna/`](../tcga_pancan_cna/README.md):
`CNA_ATTN_BLOCKS`, `TRAIN_DATA`, `VALID_DATA`, `OUTPUT_DIR`, `LOSS_DIR`,
`ATTN_BLOCKS_TO_INCLUDE`, `MATPLOTLIB_BACKEND`.

## Outputs

| Directory | Contents | Tracked? |
|---|---|---|
| `models/<model_name>/` | Trained TESSERA CNA NoLOH model artefacts | no |
| `cna_features/<model_name>_cna_features.pkl` | Per-segment latent features (train + valid) | no |
| `cna_loss/<model_name>_cna_metrics.pkl` | Predicted vs actual segment-mean + summary metrics (MSE / MAE / R^2 / Pearson). No LOH outputs. | no |
| `plots/cna_analysis/` | Predicted-vs-actual scatter (PNG) | no |

All large output directories are gitignored. Re-running the pipeline on
the same Stage-2 CSVs reproduces them deterministically.

## Compute requirements

Models are trained on an NVIDIA RTX 6000 Ada.
