"""Pretrain a TESSERA SNV model on the TCGA Pan-Cancer cohort.

Loads the TCGA training and validation SNV tables (produced by
``scripts/data/tcga/create_training_snv.py``), builds a TESSERA SNV
model under one of three architecture configurations
(``baseline`` / ``local`` / ``global``), and trains with masked-token
reconstruction. Used to produce the seven trained models reported in
Figure 1: baseline (no attention) and local + global at three flanking-
sequence context lengths each (1, 10, 25 bp).

Inputs
------
``../data/tcga/train_data_snv.csv``
``../data/tcga/valid_data_snv.csv``

Outputs
-------
Trained model artefacts written to ``models/<model_name>/`` (handled
by ``TESSERA.train_model``). The downstream scripts
(``get_variant_features.py``, ``get_variant_loss_acc.py``,
``get_variant_self_attn.py``) read these artefacts back.

Usage
-----
    python fit_model.py                                   # defaults: baseline, context_len=1
    CONFIG=local CONTEXT_LEN=25 python fit_model.py       # local, 25 bp
    CONFIG=global CONTEXT_LEN=1 python fit_model.py       # global, 1 bp
"""

import importlib
import os

import pandas as pd

from tessera.model import TESSERA

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
CONFIG_NAME = os.environ.get("CONFIG", "baseline")          # baseline | local | global
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "1"))
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_snv.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_snv.csv")

cfg = importlib.import_module(f"model_config_{CONFIG_NAME}")
MODEL_NAME = ("TCGA_PanCan_SNV_baseline" if CONFIG_NAME == "baseline"
              else f"TCGA_PanCan_SNV_{CONFIG_NAME}_{CONTEXT_LEN}")

# Architecture topology shared across the three configs (only the attention
# toggles in model_config_*.py differ).
LOCAL_CONV_DIM = [256]
LOCAL_CONV_KERNEL = [10]
REF_ALT_DIM = 48
LOCAL_NUM_HEADS = 12
LOCAL_DIM_PER_HEAD = 12
LOCAL_ATTENTION_BLOCKS = 3
GLOBAL_NUM_HEADS = 12
GLOBAL_DIM_PER_HEAD = 12
GLOBAL_ATTENTION_BLOCKS = 3

# Optimizer schedule.
MAX_LEARNING_RATE = 2e-4
WARMUP_STEPS = 1200
MIN_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.1

# ============================================================================
# Load data
# ============================================================================
print(f"Loading {TRAIN_DATA_PATH}")
train_data = pd.read_csv(TRAIN_DATA_PATH)
print(f"Loading {VALID_DATA_PATH}")
valid_data = pd.read_csv(VALID_DATA_PATH)

train_sample_ids = train_data["Tumor_Sample_Barcode"].values
train_chr = train_data["Chromosome"].values
train_pos = train_data["Start_Position"].values
train_ref = train_data["Reference_Allele"].values
train_alt = train_data["Tumor_Seq_Allele2"].values
train_vaf = train_data["vaf"].values

valid_sample_ids = valid_data["Tumor_Sample_Barcode"].values
valid_chr = valid_data["Chromosome"].values
valid_pos = valid_data["Start_Position"].values
valid_ref = valid_data["Reference_Allele"].values
valid_alt = valid_data["Tumor_Seq_Allele2"].values
valid_vaf = valid_data["vaf"].values

# ============================================================================
# Build model
# ============================================================================
model = TESSERA(name=MODEL_NAME, use_distributed=cfg.use_distributed,
                jit_compile=True, mixed_precision=False)

# SNV models use single-base ref/alt; longer alleles are truncated upstream.
ref_len = alt_len = 1 if cfg.mut_type == "SNV" else 10

model.create_sample_dataset(
    train_sample_ids, train_chr, train_pos, train_ref, train_alt,
    vaf=train_vaf, name="train_dataset",
    context_len=CONTEXT_LEN, batch_size=cfg.batch_size,
    is_training=True, subsample=cfg.subsample, fixed_bag_size=True,
    ref_len=ref_len, alt_len=alt_len,
)
model.create_sample_dataset(
    valid_sample_ids, valid_chr, valid_pos, valid_ref, valid_alt,
    vaf=valid_vaf, name="valid_dataset",
    context_len=CONTEXT_LEN, batch_size=cfg.batch_size,
    is_training=True, subsample=cfg.subsample, fixed_bag_size=True,
    ref_len=ref_len, alt_len=alt_len,
)

local_n_dim = LOCAL_NUM_HEADS * LOCAL_DIM_PER_HEAD
global_n_dim = GLOBAL_NUM_HEADS * GLOBAL_DIM_PER_HEAD
model.build_model(
    local_conv_dim=LOCAL_CONV_DIM, local_conv_kernel=LOCAL_CONV_KERNEL,
    ref_alt_dim=REF_ALT_DIM,
    local_embed_dim=local_n_dim, global_embed_dim=global_n_dim,
    local_num_heads=LOCAL_NUM_HEADS,
    local_ff_dim=local_n_dim * 2,
    local_attention_blocks=LOCAL_ATTENTION_BLOCKS,
    global_num_heads=GLOBAL_NUM_HEADS,
    global_ff_dim=global_n_dim * 2,
    global_attention_blocks=GLOBAL_ATTENTION_BLOCKS,
    max_learning_rate=MAX_LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    min_learning_rate=MIN_LEARNING_RATE,
    use_cosine_schedule=False,
    attention_activation_type="softmax",
    variant_local_attention=cfg.variant_local_attention,
    variant_self_attention=cfg.variant_self_attention,
    loss_non_zero_only=False,
    accuracy_non_zero_only=False,
    hinge_loss_t=None,
    log_gradients=False,
    recon_ref=cfg.recon_ref,
    dropout_rate=DROPOUT_RATE,
    attention_l1_factor=0.0,
    per_sample_loss=False,
)
print(f"Number of parameters in the model: {model.model_inf.count_params():,}")

# ============================================================================
# Train
# ============================================================================
model.train_model(
    epochs=cfg.epochs, steps_per_epoch=cfg.steps_per_epoch,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    epochs_min=cfg.epochs_min,
    validation_steps=cfg.validation_steps,
    validation_freq=cfg.validation_freq,
    reset_optimizer_epoch=None,
)
