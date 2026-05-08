# TCGA SNV pretraining and analysis

Trains the seven TESSERA SNV models reported in Figure 1 of the manuscript
and extracts the per-variant features, masked-token reconstruction loss,
and inter-variant attention matrices that downstream figure scripts
consume.

## Pipeline

```
scripts/data/tcga/train_data_snv.csv
scripts/data/tcga/valid_data_snv.csv     (model-ready CSVs from upstream)
                       │
                       ▼
                  fit_model.py    (7 models trained)
                       │
       ┌───────────────┼───────────────────┐
       ▼               ▼                   ▼
get_variant_     get_variant_      get_variant_
features.py      loss_acc.py       self_attn.py
       │               │                   │
       ▼               ▼                   ▼
var_features/    var_loss/          var_attn/
                       │
                       ▼
                plot_accuracy.py   (Figure 1 b-g, Supplementary Fig. 1-2)
```

The two upstream CSVs are produced by the data-prep pipeline at
[`scripts/data/tcga/`](../data/tcga/README.md), which itself reads the raw
TCGA Pan-Cancer release files documented in
[`data/TCGA_PanCan/`](../../data/TCGA_PanCan/README.md). Clinical metadata
(tumour type, etc.) is already baked into the train/valid CSVs by that
upstream stage, so the scripts here never read `clinical.csv` directly.

## Configurations

The three model architectures compared in Figure 1, switched via
`--config <name>`:

| Config | Local attention | Global self-attention | Default `context_len` |
|---|---|---|---|
| `baseline` | no | no | 1 bp |
| `local` | yes | no | 25 bp |
| `global` | yes | yes | 25 bp |

`context_len` is overridable on the CLI; the manuscript reports each of
`local` and `global` at three flanking-sequence widths (1, 10, 25 bp) plus
a single baseline run.

## Running

Configuration is set at the top of each script as module-level constants that
default to env-var lookups (e.g. `CONFIG_NAME = os.environ.get("CONFIG", "baseline")`),
so the scripts open cleanly in IPython / Jupyter / pdb. Edit the constants
or set env vars to override.

```bash
# Train all 7 models reported in Figure 1
./run_all_configs.sh

# After training, extract features / loss / inter-variant attention
./run_get_features.sh
./run_get_loss_acc.sh
CONFIG=global CONTEXT_LEN=25 python get_variant_self_attn.py   # COAD by default

# Generate Figure 1 b-g + supplementary figures
python plot_accuracy.py
```

Or run individual configurations:

```bash
python fit_model.py                                            # baseline / 1 (defaults)
CONFIG=local CONTEXT_LEN=25 python fit_model.py
CONFIG=global CONTEXT_LEN=25 python fit_model.py

CONFIG=local CONTEXT_LEN=25 python get_variant_features.py
CONFIG=global CONTEXT_LEN=1 python get_variant_loss_acc.py
CONFIG=global CONTEXT_LEN=25 TUMOR_TYPES="COAD READ" python get_variant_self_attn.py
```

The shell-script entry points accept custom run subsets (each `config:context_len` pair):

```bash
./run_all_configs.sh global:25            # train one model
./run_get_features.sh local:1 local:10    # extract features for two models
```

### Recognised env vars

| Variable | Used in | Notes |
|---|---|---|
| `CONFIG` | all 4 model scripts | `baseline` / `local` / `global` |
| `CONTEXT_LEN` | all 4 model scripts | bp on each side of the variant |
| `TUMOR_TYPES` | `get_variant_self_attn.py` | whitespace-separated tumour-type codes |
| `TRAIN_DATA`, `VALID_DATA` | all scripts | override default `../data/tcga/...` paths |
| `OUTPUT_DIR` | get_variant_*.py, plot_accuracy.py | override default `var_features` / `var_loss` / `var_attn` / `plots/...` |
| `LOSS_DIR` | plot_accuracy.py | input directory of `*_loss_variant.pkl` files |
| `USE_BOOTSTRAP` | plot_accuracy.py | set to `0` to skip the 95% CI bootstrap |
| `N_BOOTSTRAP` | plot_accuracy.py | bootstrap iterations (default 1000) |
| `MATPLOTLIB_BACKEND` | plot_accuracy.py | e.g. `Agg` for headless |

## Outputs

| Directory | Contents | Tracked? |
|---|---|---|
| `models/<model_name>/` | Trained TESSERA model artefacts (Keras + checkpoints + training_log.csv) | no |
| `var_features/<model_name>_features.pkl` | Per-variant latent features (train + valid) | no |
| `var_loss/<model_name>_loss_variant.pkl` | Per-variant masked-token loss + ref/alt logits + ground truth (train + valid) | no |
| `var_attn/<model_name>/<tumor_types>.pkl` | Per-sample inter-variant attention matrices joined with variant metadata | no |
| `plots/accuracy_analysis/` | Generated manuscript figures (PDF + PNG) | no |

All large model + features + plot directories are gitignored. A user
re-running the pipeline regenerates them deterministically (random seeds
are pinned).

## Manuscript mapping

- Figure 1 b-c, e-g (per-variant masked-token accuracy across architectures
  / tumor types / variant burden) -> `plot_accuracy.py`
- Figure 1 d (TCGA tumor-type accuracy heatmap) -> `plot_accuracy.py`
- Figure 1 f-g (GENIE cross-platform transfer) -> uses `var_loss/` files but
  is generated by the GENIE-side script in `scripts/genie_snv/`
- Figure 1 h-o (ClinVar variant-pathogenicity prediction) -> uses
  `var_features/` files; downstream scripts live elsewhere
- Inter-variant attention exploration in supplementary -> uses
  `var_attn/<global_25>/COAD.pkl`

## Compute requirements

Models are trained on an NVIDIA RTX 6000 Ada.
