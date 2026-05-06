"""TESSERA CNA-only model configuration (NoLOH variant).

The NoLOH variant trains the CNA encoder with only the segment-mean
regression head — no allele-specific copy-number / LOH classification.
This is the variant used for cross-platform inference on panel-sequencing
cohorts (MSK-CHORD, Figure 2 d) where allele-specific copy-number is
unavailable.

Used by ``fit_model.py``, ``get_cna_features.py``, and
``get_cna_loss_metrics.py``. The number of inter-segment self-attention
blocks is set on the env-var ``CNA_ATTN_BLOCKS`` in each script; the model
name is derived from ``TCGA_PanCan_CNA_NoLOH_attn_<N>``.
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

# Default model-name base. Per-config name is f"{model_name}_attn_{cna_attention_blocks}".
model_name = "TCGA_PanCan_CNA_NoLOH"

# LOH classification head is disabled in the NoLOH variant.
predict_cna_loh = False

# Segment-mean normalisation (applied at dataset-creation time).
z_score_cna = False
z_score_clip = None
