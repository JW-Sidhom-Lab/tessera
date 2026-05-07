# GENIE SNV cross-platform validation (Figure 1 f-g)

Runs the seven TCGA-pretrained SNV models (baseline + local_{1, 10, 25}
+ global_{1, 10, 25}) on the AACR Project GENIE v18.0 panel-sequencing
SNV cohort. Computes per-variant ALT and REF allele predictions, logits,
and loss; aggregates them into per-cohort and per-variant-burden
accuracy panels backing Fig. 1 f-g.

## Pipeline

```
../data/genie/snv.csv                       (panel-restricted GENIE SNVs from scripts/data/genie/)
../tcga_pancan_snv/models/<MODEL>/          (TCGA-pretrained SNV checkpoints)
                       │
                       ▼ get_variant_loss_acc.py [MODEL_NAME]
       var_loss/<MODEL>_genie_loss_variant.pkl
                       │
                       ▼ plot_accuracy.py
       plots/accuracy_analysis/*.png
```

`run_inference.sh` iterates over the seven manuscript models and calls
`get_variant_loss_acc.py` once per model.

## Files

| Script | What it does |
|---|---|
| [`get_variant_loss_acc.py`](get_variant_loss_acc.py) | Frozen TESSERA inference on the GENIE SNV table for one TCGA-pretrained model; emits a per-model `var_loss/*.pkl` with predictions, logits, loss, and the filtered GENIE rows. |
| [`plot_accuracy.py`](plot_accuracy.py) | Loads every `var_loss/*.pkl` and renders the per-sequence-type, per-variant-burden, and per-cancer-type accuracy panels (600 dpi PNG, bootstrap 95% CIs). |
| [`run_inference.sh`](run_inference.sh) | Driver that runs `get_variant_loss_acc.py` for each of the seven manuscript SNV models in turn. |

## Running it

```bash
cd scripts/genie_snv

# 1. Frozen-model inference on GENIE for all 7 SNV models (GPU recommended)
./run_inference.sh

# 2. Render the accuracy panels
python3 plot_accuracy.py
```

Per-model overrides:

```bash
python3 get_variant_loss_acc.py TCGA_PanCan_SNV_global_25
```
