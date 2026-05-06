"""Pretrain a TESSERA CNA-only NoLOH model on the TCGA Pan-Cancer cohort.

Trains the CNA encoder with only the segment-mean regression head, no
allele-specific LOH classification. Used to produce the NoLOH model that
the manuscript transfers to MSK-CHORD panel sequencing for the CNA cross-
platform validation (Figure 2 d), where allele-specific copy-number is
unavailable.

Inputs
------
``../data/tcga/train_data_cna.csv``
``../data/tcga/valid_data_cna.csv``

Outputs
-------
Trained model artefacts in ``models/<model_name>/``. Downstream scripts
(``get_cna_features.py``, ``get_cna_loss_metrics.py``) read these back.

Usage
-----
    python fit_model.py                              # defaults: attn_2
    CNA_ATTN_BLOCKS=1 python fit_model.py            # attn_1
    CNA_ATTN_BLOCKS=0 python fit_model.py            # attn_0
"""

import os

import pandas as pd

from tessera.model import TESSERA

import model_config

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
# Default to attn_2 since that is the variant used for cross-platform
# validation in the manuscript; CNA_ATTN_BLOCKS overrides if set.
CNA_ATTN_BLOCKS = int(os.environ.get("CNA_ATTN_BLOCKS", "2"))
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_cna.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_cna.csv")

MODEL_NAME = f"{model_config.model_name}_attn_{CNA_ATTN_BLOCKS}"

# Architecture topology.
CNA_NUM_HEADS = 12
DIM_PER_HEAD_CNA = 12
CNA_EMBED_DIM = CNA_NUM_HEADS * DIM_PER_HEAD_CNA
CNA_FF_DIM = CNA_EMBED_DIM * 2
CHR_EMBED_DIM = 16

# Optimizer schedule.
MAX_LEARNING_RATE = 2e-4
WARMUP_STEPS = 1200
MIN_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.1

# ============================================================================
# Load data (LOH input deliberately not consumed in the NoLOH variant)
# ============================================================================
print(f"Loading {TRAIN_DATA_PATH}")
train_data_cna = pd.read_csv(TRAIN_DATA_PATH)
print(f"Loading {VALID_DATA_PATH}")
valid_data_cna = pd.read_csv(VALID_DATA_PATH)
print(f"Training:   {len(train_data_cna):,} segments / {train_data_cna['Tumor_Sample_Barcode'].nunique():,} samples")
print(f"Validation: {len(valid_data_cna):,} segments / {valid_data_cna['Tumor_Sample_Barcode'].nunique():,} samples")

train_cna_sample_ids = train_data_cna["Tumor_Sample_Barcode"].values
train_cna_chr = train_data_cna["Chromosome"].values
train_cna_start = train_data_cna["Start"].astype(int).values
train_cna_end = train_data_cna["End"].astype(int).values
train_cna_segment_mean = train_data_cna["Segment_Mean"].astype(float).values

valid_cna_sample_ids = valid_data_cna["Tumor_Sample_Barcode"].values
valid_cna_chr = valid_data_cna["Chromosome"].values
valid_cna_start = valid_data_cna["Start"].astype(int).values
valid_cna_end = valid_data_cna["End"].astype(int).values
valid_cna_segment_mean = valid_data_cna["Segment_Mean"].astype(float).values

# ============================================================================
# Build model
# ============================================================================
model = TESSERA(name=MODEL_NAME, use_distributed=model_config.use_distributed,
                jit_compile=True, mixed_precision=False)

dataset_kwargs = dict(
    cna_subsample=model_config.cna_subsample,
    batch_size=model_config.batch_size,
    is_training=True, fixed_bag_size=True,
    z_score_cna=model_config.z_score_cna,
    z_score_clip=model_config.z_score_clip,
)
model.create_sample_dataset(
    cna_sample_ids=train_cna_sample_ids, cna_chromosomes=train_cna_chr,
    cna_starts=train_cna_start, cna_ends=train_cna_end,
    cna_segment_means=train_cna_segment_mean, cna_lohs=None,
    name="train_dataset", **dataset_kwargs,
)
model.create_sample_dataset(
    cna_sample_ids=valid_cna_sample_ids, cna_chromosomes=valid_cna_chr,
    cna_starts=valid_cna_start, cna_ends=valid_cna_end,
    cna_segment_means=valid_cna_segment_mean, cna_lohs=None,
    name="valid_dataset", **dataset_kwargs,
)

model.build_model(
    cna_embed_dim=CNA_EMBED_DIM,
    cna_num_heads=CNA_NUM_HEADS,
    cna_ff_dim=CNA_FF_DIM,
    cna_attention_blocks=CNA_ATTN_BLOCKS,
    use_chr_one_hot=True,
    chr_embed_dim=CHR_EMBED_DIM,

    train_on_mutation_loss=False,
    train_on_cna_loss=True,
    loss_non_zero_only=False,
    predict_cna_loh=model_config.predict_cna_loh,
    accuracy_non_zero_only=False,
    hinge_loss_t=None,
    per_sample_loss=False,

    max_learning_rate=MAX_LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    min_learning_rate=MIN_LEARNING_RATE,
    use_cosine_schedule=False,

    dropout_rate=DROPOUT_RATE,
    attention_activation_type="softmax",
    attention_l1_factor=0.0,
    log_gradients=False,
)
print(f"Number of parameters in the model: {model.model_inf.count_params():,}")

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
    reset_optimizer_epoch=None,
)
