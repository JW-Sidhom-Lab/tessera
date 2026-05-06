# MSK-CHORD CNA cross-platform validation

Reproduces Figure 2 d of the manuscript: applies the TCGA-trained
TESSERA CNA NoLOH model (from
[`scripts/tcga_pancan_cna_noloh/`](../tcga_pancan_cna_noloh/README.md))
to MSK-CHORD panel-sequenced CNA segments and reports per-segment
Pearson correlation between predicted and actual Segment_Mean as the
cross-platform-transfer benchmark.

## Pipeline

```
scripts/data/msk_chord/cna_panel_filtered.csv
                      │
                      ▼
       (optional) compare_cna_distributions.py    diagnostic figure: TCGA vs MSK distribution shift
                      │
                      ▼
       get_cna_loss_acc.py    quantile-remaps MSK Segment_Mean onto TCGA
                              distribution, runs trained NoLOH attn_<N> model
                      │
                      ▼
       cna_loss_panel_filtered/<MODEL>_msk_chord_loss_cna.pkl
                      │
                      ▼
       plot_cna_reconstruction.py    Figure 2 d
```

## Running

```bash
# (optional) diagnostic distribution comparison
python compare_cna_distributions.py

# Cross-platform inference for all 3 attention configs + Figure 2 d
./run_inference.sh

# Or one config at a time
CNA_ATTN_BLOCKS=2 python get_cna_loss_acc.py
python plot_cna_reconstruction.py
```

`get_cna_loss_acc.py` reads:

- panel-filtered MSK-CHORD CNA segments from
  [`scripts/data/msk_chord/`](../data/msk_chord/README.md) (produced by
  `create_cna.py` there)
- the trained NoLOH model from
  [`scripts/tcga_pancan_cna_noloh/`](../tcga_pancan_cna_noloh/README.md)
- the TCGA Segment_Mean reference distribution (lazily cached at
  `../data/tcga/cna_stats.json` and `../data/tcga/cna_sorted.npy`,
  computed on first use from
  `scripts/data/tcga/{train,valid}_data_cna.csv`)

so the upstream NoLOH pretraining and MSK-CHORD CNA pipelines must have
run before this analysis.

### Recognised env vars

| Variable | Used in | Notes |
|---|---|---|
| `CNA_ATTN_BLOCKS` | `get_cna_loss_acc.py` | `0` / `1` / `2` (default `2`, the manuscript variant) |
| `MODEL_NAME` | `get_cna_loss_acc.py` | full override of the derived model name |
| `MODELS_DIR` | `get_cna_loss_acc.py` | parent dir for the trained model (default `../tcga_pancan_cna_noloh/models`) |
| `DATA_SOURCE` | `get_cna_loss_acc.py` | `panel_filtered` (default) or `raw` (full segments) |
| `NORMALIZATION_MODE` | `get_cna_loss_acc.py` | `quantile` (default), `linear`, or `none` |
| `N_SAMPLES` | `get_cna_loss_acc.py` | optional cap on number of MSK samples scored |
| `OUTPUT_DIR` | all scripts | override default output dir |
| `LOSS_DIR` | `plot_cna_reconstruction.py` | input directory of `*_msk_chord_loss_cna.pkl` |
| `ATTN_BLOCKS_TO_INCLUDE` | `plot_cna_reconstruction.py` | comma-separated list, default `0,1,2` |
| `Z_SCORE_NORM`, `RAW_PLOT_LIM` | `plot_cna_reconstruction.py` | axis-style flags |
| `MATPLOTLIB_BACKEND` | plot scripts | e.g. `Agg` for headless |

## Outputs

| Path | Description |
|---|---|
| `cna_loss_panel_filtered/<MODEL>_msk_chord_loss_cna.pkl` | Per-model predictions + summary metrics on MSK-CHORD |
| `plots/cna_distribution_comparison.png` + `cna_distribution_summary.csv` | Diagnostic: TCGA vs MSK Segment_Mean distributions across normalisation modes |
| `plots/cna_analysis/cna_predicted_vs_actual.png` | Figure 2 d |

The pipeline is deterministic given fixed upstream inputs.

## Compute requirements

Inference runs on an NVIDIA RTX 6000 Ada.
