"""Pretrain the joint TESSERA SNV+CNA model on the TCGA Pan-Cancer cohort.

Loads the TCGA training and validation SNV + CNA tables (produced by
``scripts/data/tcga/create_training_snv.py`` and
``create_training_cna.py``) and trains a dual-branch TESSERA model
under ``model_config_infonce.py``: SNV branch (local + global
attention), CNA branch (segment self-attention), masked-token
reconstruction on both modalities, and a per-sample InfoNCE alignment
loss between the pooled SNV and CNA sample embeddings. The two
encoders are coupled only through the contrastive objective; the
published config sets ``cross_modal_blocks=0``, so no cross-modal
attention block is built.

Inputs
------
``../data/tcga/train_data_snv.csv``
``../data/tcga/valid_data_snv.csv``
``../data/tcga/train_data_cna.csv``
``../data/tcga/valid_data_cna.csv``

Outputs
-------
Trained model artefacts in ``models/<model_name>/``. Downstream scripts
(``get_cna_variant_features.py``, ``get_snv_cna_loss_metrics.py``)
read these back.

Usage
-----
    python fit_model.py
"""

import os

import pandas as pd

from tessera.model import TESSERA

import model_config_infonce as model_config

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
TRAIN_SNV_PATH = os.environ.get("TRAIN_SNV", "../data/tcga/train_data_snv.csv")
VALID_SNV_PATH = os.environ.get("VALID_SNV", "../data/tcga/valid_data_snv.csv")
TRAIN_CNA_PATH = os.environ.get("TRAIN_CNA", "../data/tcga/train_data_cna.csv")
VALID_CNA_PATH = os.environ.get("VALID_CNA", "../data/tcga/valid_data_cna.csv")

# Architecture topology (matches the SNV-only and CNA-only encoders).
LOCAL_CONV_DIM = [256]
LOCAL_CONV_KERNEL = [10]
REF_ALT_DIM = 48
LOCAL_NUM_HEADS = 12
LOCAL_DIM_PER_HEAD = 12
LOCAL_ATTENTION_BLOCKS = 3
GLOBAL_NUM_HEADS = 12
GLOBAL_DIM_PER_HEAD = 12
GLOBAL_ATTENTION_BLOCKS = 3
CNA_NUM_HEADS = 12
DIM_PER_HEAD_CNA = 12
CHR_EMBED_DIM = 16
CROSS_MODAL_NUM_HEADS = 12
DIM_PER_HEAD_CROSS = 12

LOCAL_EMBED_DIM = LOCAL_NUM_HEADS * LOCAL_DIM_PER_HEAD
LOCAL_FF_DIM = LOCAL_EMBED_DIM * 2
GLOBAL_EMBED_DIM = GLOBAL_NUM_HEADS * GLOBAL_DIM_PER_HEAD
GLOBAL_FF_DIM = GLOBAL_EMBED_DIM * 2
CNA_EMBED_DIM = CNA_NUM_HEADS * DIM_PER_HEAD_CNA
CNA_FF_DIM = CNA_EMBED_DIM * 2
CROSS_MODAL_EMBED_DIM = CROSS_MODAL_NUM_HEADS * DIM_PER_HEAD_CROSS
CROSS_MODAL_FF_DIM = CROSS_MODAL_EMBED_DIM * 2

# Optimizer + regularisation.
MAX_LEARNING_RATE = 2e-4
WARMUP_STEPS = 1200
MIN_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.1
TOKEN_BAG_LOSS_WEIGHT = 0.01

# ============================================================================
# Load data
# ============================================================================
print(f"Loading SNV: {TRAIN_SNV_PATH}, {VALID_SNV_PATH}")
train_data_snv = pd.read_csv(TRAIN_SNV_PATH)
valid_data_snv = pd.read_csv(VALID_SNV_PATH)
print(f"  Training:   {len(train_data_snv):,} variants / "
      f"{train_data_snv['Tumor_Sample_Barcode'].nunique():,} samples")
print(f"  Validation: {len(valid_data_snv):,} variants / "
      f"{valid_data_snv['Tumor_Sample_Barcode'].nunique():,} samples")

print(f"Loading CNA: {TRAIN_CNA_PATH}, {VALID_CNA_PATH}")
train_data_cna = pd.read_csv(TRAIN_CNA_PATH)
valid_data_cna = pd.read_csv(VALID_CNA_PATH)
print(f"  Training:   {len(train_data_cna):,} segments / "
      f"{train_data_cna['Tumor_Sample_Barcode'].nunique():,} samples")
print(f"  Validation: {len(valid_data_cna):,} segments / "
      f"{valid_data_cna['Tumor_Sample_Barcode'].nunique():,} samples")

# SNV columns
train_sample_ids = train_data_snv["Tumor_Sample_Barcode"].values
train_chr = train_data_snv["Chromosome"].astype(str).values
train_pos = train_data_snv["Start_Position"].astype(int).values
train_ref = train_data_snv["Reference_Allele"].values
train_alt = train_data_snv["Tumor_Seq_Allele2"].values
train_vaf = train_data_snv["vaf"].values

valid_sample_ids = valid_data_snv["Tumor_Sample_Barcode"].values
valid_chr = valid_data_snv["Chromosome"].astype(str).values
valid_pos = valid_data_snv["Start_Position"].astype(int).values
valid_ref = valid_data_snv["Reference_Allele"].values
valid_alt = valid_data_snv["Tumor_Seq_Allele2"].values
valid_vaf = valid_data_snv["vaf"].values

# CNA columns. Only read LOH when we'll use it; otherwise pass None so the
# model is built without a cna_loh input tensor (clean no-LOH training).
train_cna_sample_ids = train_data_cna["Tumor_Sample_Barcode"].values
train_cna_chr = train_data_cna["Chromosome"].astype(str).values
train_cna_start = train_data_cna["Start"].astype(int).values
train_cna_end = train_data_cna["End"].astype(int).values
train_cna_segment_mean = train_data_cna["Segment_Mean"].astype(float).values
train_cna_loh = (train_data_cna["LOH"].astype(bool).values
                 if model_config.predict_cna_loh else None)

valid_cna_sample_ids = valid_data_cna["Tumor_Sample_Barcode"].values
valid_cna_chr = valid_data_cna["Chromosome"].astype(str).values
valid_cna_start = valid_data_cna["Start"].astype(int).values
valid_cna_end = valid_data_cna["End"].astype(int).values
valid_cna_segment_mean = valid_data_cna["Segment_Mean"].astype(float).values
valid_cna_loh = (valid_data_cna["LOH"].astype(bool).values
                 if model_config.predict_cna_loh else None)

# ============================================================================
# Build model + datasets
# ============================================================================
model = TESSERA(name=model_config.model_name,
                use_distributed=model_config.use_distributed,
                jit_compile=True, mixed_precision=False)

ref_len = alt_len = 1 if model_config.mut_type == "SNV" else 10

dataset_kwargs_common = dict(
    context_len=model_config.context_len,
    batch_size=model_config.batch_size,
    is_training=True,
    subsample=model_config.subsample,
    fixed_bag_size=True,
    ref_len=ref_len,
    alt_len=alt_len,
    cna_subsample=model_config.cna_subsample,
    precompute=True,
    z_score_cna=model_config.z_score_cna,
    z_score_clip=model_config.z_score_clip,
)

model.create_sample_dataset(
    sample_ids=train_sample_ids, chromosomes=train_chr, positions=train_pos,
    refs=train_ref, alts=train_alt, vaf=train_vaf,
    cna_sample_ids=train_cna_sample_ids, cna_chromosomes=train_cna_chr,
    cna_starts=train_cna_start, cna_ends=train_cna_end,
    cna_segment_means=train_cna_segment_mean, cna_lohs=train_cna_loh,
    name="train_dataset",
    **dataset_kwargs_common,
)
model.create_sample_dataset(
    sample_ids=valid_sample_ids, chromosomes=valid_chr, positions=valid_pos,
    refs=valid_ref, alts=valid_alt, vaf=valid_vaf,
    cna_sample_ids=valid_cna_sample_ids, cna_chromosomes=valid_cna_chr,
    cna_starts=valid_cna_start, cna_ends=valid_cna_end,
    cna_segment_means=valid_cna_segment_mean, cna_lohs=valid_cna_loh,
    name="valid_dataset",
    precompute_batches=model_config.validation_steps,
    **dataset_kwargs_common,
)

model.build_model(
    recon_ref=model_config.recon_ref,
    # Mutation local
    local_conv_dim=LOCAL_CONV_DIM, local_conv_kernel=LOCAL_CONV_KERNEL,
    ref_alt_dim=REF_ALT_DIM,
    local_embed_dim=LOCAL_EMBED_DIM, local_num_heads=LOCAL_NUM_HEADS,
    local_ff_dim=LOCAL_FF_DIM, local_attention_blocks=LOCAL_ATTENTION_BLOCKS,
    # Mutation global
    global_embed_dim=GLOBAL_EMBED_DIM, global_num_heads=GLOBAL_NUM_HEADS,
    global_ff_dim=GLOBAL_FF_DIM, global_attention_blocks=GLOBAL_ATTENTION_BLOCKS,
    # Mutation attention flags
    variant_local_attention=model_config.variant_local_attention,
    variant_self_attention=model_config.variant_self_attention,
    # CNA
    cna_embed_dim=CNA_EMBED_DIM, cna_num_heads=CNA_NUM_HEADS,
    cna_ff_dim=CNA_FF_DIM, cna_attention_blocks=model_config.cna_attention_blocks,
    use_chr_one_hot=True, chr_embed_dim=CHR_EMBED_DIM,
    # Cross-modal
    cross_modal_num_heads=CROSS_MODAL_NUM_HEADS,
    cross_modal_ff_dim=CROSS_MODAL_FF_DIM,
    cross_modal_blocks=model_config.cross_modal_blocks,
    cross_attention_explicit=model_config.cross_attention_explicit,
    # Loss config
    train_on_mutation_loss=model_config.train_on_mutation_loss,
    train_on_cna_loss=model_config.train_on_cna_loss,
    predict_cna_loh=model_config.predict_cna_loh,
    loss_non_zero_only=False, accuracy_non_zero_only=False,
    hinge_loss_t=None,
    per_sample_loss=model_config.per_sample_loss,
    use_infonce_loss=model_config.use_infonce_loss,
    infonce_loss_weight=model_config.infonce_loss_weight,
    infonce_projection_dim=model_config.infonce_projection_dim,
    infonce_shared_projection=model_config.infonce_shared_projection,
    infonce_projection_n_layers=model_config.infonce_projection_n_layers,
    infonce_projection_dropout=getattr(model_config, "infonce_projection_dropout", None),
    use_mut_token_bag_infonce=model_config.use_mut_token_bag_infonce,
    use_cna_token_bag_infonce=model_config.use_cna_token_bag_infonce,
    token_bag_loss_weight=TOKEN_BAG_LOSS_WEIGHT,
    # Optimizer schedule
    max_learning_rate=MAX_LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    min_learning_rate=MIN_LEARNING_RATE,
    use_cosine_schedule=False,
    # Regularization
    dropout_rate=DROPOUT_RATE,
    attention_activation_type="softmax",
    attention_l1_factor=0.0,
    log_gradients=False,
)

print(f"Model built. Total parameters: {model.model_inf.count_params():,}")

# ============================================================================
# Train
# ============================================================================
model.train_model(
    epochs=model_config.epochs,
    steps_per_epoch=model_config.steps_per_epoch,
    validation_steps=model_config.validation_steps,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    epochs_min=model_config.epochs_min,
    validation_freq=model_config.validation_freq,
    early_stopping_min_relative_delta=model_config.early_stopping_min_relative_delta,
)
print(f"Training complete. Artefacts saved to {model.model_dir}")
