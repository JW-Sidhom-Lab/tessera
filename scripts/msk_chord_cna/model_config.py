"""TESSERA CNA NoLOH model configuration for MSK-CHORD cross-platform inference.

The cross-platform validation in Figure 2 d transfers the TCGA-trained
CNA NoLOH model (from ``scripts/tcga_pancan_cna_noloh/``) to MSK-CHORD
panel-sequenced CNA segments. MSK-CHORD lacks allele-specific copy-
number, so we use the NoLOH variant.

Used by ``get_cna_loss_acc.py``. The number of attention blocks is set
on the ``CNA_ATTN_BLOCKS`` env var; the model name is derived from
``TCGA_PanCan_CNA_NoLOH_attn_<N>``.
"""

# Inference batch size.
batch_size = 24

# Sample-bag construction.
cna_subsample = None       # use all CNA segments
use_distributed = False
model_name = "TCGA_PanCan_CNA_NoLOH"

# Segment-mean normalisation applied at dataset-creation time (kept off
# here; the manuscript pre-normalises the panel segment-mean distribution
# onto TCGA's via the quantile remap below before this stage).
z_score_cna = False
z_score_clip = None
