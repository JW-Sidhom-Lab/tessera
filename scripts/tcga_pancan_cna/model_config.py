"""TESSERA CNA-only model configuration.

Used by ``fit_model.py``, ``get_cna_features.py``, and
``get_cna_loss_metrics.py``. The number of inter-segment self-attention
blocks (``cna_attention_blocks``) is set on the env-var ``CNA_ATTN_BLOCKS``
in each script; the model name is derived from
``TCGA_PanCan_CNA_attn_<N>``.
"""

# Training schedule.
batch_size = 24
epochs = 100
epochs_min = 15
steps_per_epoch = 7500 // batch_size
validation_steps = 2500 // batch_size
validation_freq = 1

# Sample-bag construction.
cna_subsample = None
use_distributed = False

# Default model-name base; the per-config name is built as
# f"{model_name}_attn_{cna_attention_blocks}".
model_name = "TCGA_PanCan_CNA"

# Loss configuration. ``predict_cna_loh = True`` enables the LOH
# classification head alongside the segment-mean regression. The manuscript
# reports both with-LOH and NoLOH variants; flip this in the script's
# top-level constants when reproducing the NoLOH ablation.
predict_cna_loh = True

# Segment-mean normalisation (applied at dataset-creation time).
z_score_cna = False        # True = z-score CNA features
z_score_clip = None        # Clip z-scored values to +/- this; None disables
