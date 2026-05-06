"""Configuration for the TESSERA joint SNV+CNA InfoNCE pretraining.

The single configuration reported in the manuscript: dual-branch SNV
(local + global attention) and CNA (segment self-attention) encoders
joined by an explicit cross-attention block, trained with masked-token
reconstruction on both modalities plus a per-sample InfoNCE alignment
loss between the SNV and CNA sample-level pooled embeddings.
"""

# Data configuration
mut_type = "SNV"
context_len = 25

# Training configuration
batch_size = 24
epochs = 1000
epochs_min = 15
steps_per_epoch = 7500 // batch_size
validation_steps = 2500 // batch_size
validation_freq = 1

# Model configuration
subsample = None
cna_subsample = None
use_distributed = False
model_name = "TCGA_SNV_CNA"

# Architecture flags
variant_local_attention = True
variant_self_attention = True
cna_attention_blocks = 2
recon_ref = True              # always True for multi-modal models
use_cna = True
train_on_mutation_loss = True
train_on_cna_loss = True
cross_attention_explicit = True
predict_cna_loh = False
use_mut_token_bag_infonce = False
use_cna_token_bag_infonce = False
use_infonce_loss = True
per_sample_loss = True
infonce_loss_weight = 0.1
infonce_projection_dim = 256
infonce_shared_projection = True
infonce_projection_n_layers = 2
infonce_projection_dropout = 0.1
early_stopping_min_relative_delta = 0.0
cross_modal_blocks = 0
z_score_cna = False
z_score_clip = None

# Derived model name with suffixes that match the trained-checkpoint directory
# layout. Kept as in-line conditionals so that flipping a flag yields the
# expected disk path without manual edits.
if use_infonce_loss:
    model_name += "_InfoNCE"
if per_sample_loss:
    model_name += "_per_sample_loss"
if not predict_cna_loh:
    model_name += "_noLOH"
if cross_modal_blocks != 0:
    model_name += f"_cross_modal_{cross_modal_blocks}_blocks"
