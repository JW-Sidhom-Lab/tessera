# Joint SNV+CNA InfoNCE pretraining (TESSERA multimodal foundation model)

Pretrains the joint TESSERA model on TCGA Pan-Cancer somatic mutations
and copy-number alterations together. Two parallel encoders (SNV
local+global attention; CNA segment self-attention) are joined by an
explicit cross-attention block so the two modalities can re-contextualize
each other, and the pooled per-sample embeddings are pulled together
across modalities by a per-sample InfoNCE loss. The trained encoder
feeds the InfoNCE-aligned downstream tumour-type classifier in
[`scripts/tumor_classification_snv_cna_infonce/`](../tumor_classification_snv_cna_infonce/README.md).

## Pipeline

```
scripts/data/tcga/{train,valid}_data_snv.csv
scripts/data/tcga/{train,valid}_data_cna.csv
                       │
                       ▼
              fit_model.py
                       │
                       ▼
              models/<model_name>/
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
get_cna_variant_features.py    get_snv_cna_loss_metrics.py
(per-variant + per-segment     (per-token loss + CNA Segment_Mean
 latent features)               + LOH reconstruction metrics)
       │                               │
       ▼                               ▼
multimodal_features/             multimodal_loss/
```

`<model_name>` is derived from `model_config_infonce.py` and resolves to
`TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH` for the manuscript config.

## Manuscript configuration

The single configuration in `model_config_infonce.py`:

- **SNV branch**: 3 local-attention blocks + 3 global self-attention
  blocks, 144-dim embeddings (12 heads x 12 dim).
- **CNA branch**: 2 segment self-attention blocks, 144-dim embeddings.
- **Cross-modal**: explicit cross-attention (`cross_attention_explicit
  = True`), no additional cross-modal block stack
  (`cross_modal_blocks = 0`).
- **Loss**: masked-token reconstruction on both modalities, plus
  per-sample InfoNCE alignment between the pooled SNV and CNA sample
  embeddings (InfoNCE weight 0.1, temperature 0.1).
- **Projection heads** (one per modality): a 2-layer MLP with 256
  units per layer (one GELU-activated hidden layer plus a linear
  output), applied to the per-token backbone features prior to any
  cross-modal interaction, then masked global average pooling to a
  sample-level embedding (project-then-mean). Set by
  `infonce_projection_n_layers = 2`, `infonce_projection_dim = 256`,
  `infonce_shared_projection = True`.
- **LOH head**: optional, set by `predict_cna_loh`. Both variants are
  trained and reported in the manuscript: the LOH variant
  (`predict_cna_loh = True`) carries an allele-specific copy-number
  classification head trained jointly with Segment_Mean reconstruction,
  and the NoLOH variant (`predict_cna_loh = False`) drops that head and
  is the variant required for cross-platform validation on
  panel-sequenced cohorts (e.g., MSK-CHORD) where allele-specific
  copy-number calls are not available. Toggling this flag retrains the
  other variant; the resulting `model_name` and on-disk directory are
  suffixed automatically (`..._noLOH` when `predict_cna_loh = False`).

## Missing-modality handling

Joint training does not require both SNV and CNA data per sample. Every
sample that has at least one modality enters the dataset and contributes
to that modality's reconstruction loss; only the intersection cohort
drives the cross-modal InfoNCE alignment.

- `tessera.base._prepare_dataset_inputs` collects all SNV-bearing
  samples, then unions in CNA-only samples (with empty SNV bag) so a
  single batch can mix SNV-only / CNA-only / both-modality samples.
  Missing slots are zero-padded; the chromosome embedding's `0` index
  is reserved as the padding sentinel.
- In the `train_step`, a per-sample boolean `valid_pairs_mask` is
  computed from the padding sentinel:
  `has_mut = any(chr != 0)` and `has_cna = any(cna_chr != 0)`,
  per sample; `valid_pairs_mask = has_mut AND has_cna`.
- `compute_infonce_loss` multiplies the per-sample contrastive loss by
  this mask and divides by the number of valid pairs, so single-
  modality samples contribute zero to both numerator and denominator
  of the InfoNCE term.
- The two reconstruction losses are masked at the **token** level (via
  the same padding sentinel), so a CNA-only sample's empty SNV bag
  contributes zero to the SNV reconstruction loss while its CNA
  segments contribute normally, and vice versa.

The net effect on the manuscript model: every TCGA Pan-Cancer sample
with either modality contributes to its own reconstruction objective,
while only the (large) SNV+CNA intersection cohort shapes the
cross-modal alignment.

## Running

```bash
# Train + extract joint features (manuscript pipeline)
./run_pipeline.sh

# Or step by step
python fit_model.py
python get_cna_variant_features.py
python get_snv_cna_loss_metrics.py    # optional: reconstruction metrics
```

Each script reads:

- SNV train/valid tables from
  [`scripts/data/tcga/`](../data/tcga/README.md)
  (`train_data_snv.csv`, `valid_data_snv.csv`)
- CNA train/valid tables from the same directory
  (`train_data_cna.csv`, `valid_data_cna.csv`)
- For the two extraction scripts: the trained-model artefacts written
  by `fit_model.py` under `models/<model_name>/`.

so the upstream TCGA data preparation must have run first.

### Recognised env vars

| Variable | Used in | Notes |
|---|---|---|
| `TRAIN_SNV`, `VALID_SNV` | all | SNV table paths (default `../data/tcga/{train,valid}_data_snv.csv`) |
| `TRAIN_CNA`, `VALID_CNA` | all | CNA table paths |
| `OUTPUT_DIR` | `get_cna_variant_features.py` / `get_snv_cna_loss_metrics.py` | Output dir override (defaults `multimodal_features/`, `multimodal_loss/`) |

The active configuration lives in `model_config_infonce.py`; edit that
file (or copy it under a new name and re-import in the `fit_model.py`
import line) to explore alternative configs.

## Outputs

| Path | Description |
|---|---|
| `models/<model_name>/` | Trained encoder + cross-attention + projection heads + training log |
| `multimodal_features/<model_name>_multimodal_features.pkl` | Per-variant + per-CNA-segment latent features (train + valid) |
| `multimodal_loss/<model_name>_multimodal_metrics.pkl` | Per-token SNV loss + CNA Segment_Mean / LOH reconstruction metrics |

The two output dirs are gitignored.

## Loading legacy checkpoints

Checkpoints saved by the previous `pkg.var_llm.VariantLLM` training
stack (anything trained before the `tessera/` package migration) bake
the old import paths into their `config.json`. To load them with the
new package, run the relocator once on the source directory:

```bash
python -m tessera._legacy /path/to/models_old /path/to/models
```

Each `.keras` archive is rewritten in place: `pkg.functions.*` becomes
`tessera.*` and any unsupported `quantization_config: null` entries
(emitted by newer Keras versions on RunPod, rejected by older Keras
versions locally) are stripped. The relocated archives load under
vanilla `keras.models.load_model` against this package, with no
runtime shim required.

If you'd prefer not to rewrite the file on disk (e.g., when inspecting
an external archive), `tessera._legacy.install_legacy_aliases()`
installs `sys.modules` aliases for the old paths instead, after which
the original archive loads directly.

## Compute requirements

Trained on an NVIDIA H100 (~24 hours wall-clock for the manuscript
config at `epochs=1000` with the early-stopping schedule).
