"""Extract per-sample inter-variant attention matrices from a trained global-attention TESSERA model.

Loads the TCGA training and validation SNV tables, restricts to a tumor-
type cohort (default ``COAD``), runs the model in inference mode, and
saves the inter-variant attention matrices joined with variant metadata
to ``var_attn/<model_name>/<tumor_types>.pkl``.

Only meaningful for the ``global`` config (``variant_self_attention=True``);
the ``baseline`` and ``local`` configs have no inter-variant attention.

Inputs
------
``../data/tcga/train_data_snv.csv``
``../data/tcga/valid_data_snv.csv``

Output
------
``var_attn/<model_name>/<tumor_types>.pkl``

Usage
-----
    CONFIG=global CONTEXT_LEN=25 python get_variant_self_attn.py
    CONFIG=global CONTEXT_LEN=25 TUMOR_TYPES="COAD READ" python get_variant_self_attn.py
"""

import importlib
import os
import pickle

import pandas as pd

from tessera.data.preprocessing import create_attention_df
from tessera.model import TESSERA

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
CONFIG_NAME = os.environ.get("CONFIG", "global")            # baseline | local | global
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "25"))
TUMOR_TYPES = os.environ.get("TUMOR_TYPES", "COAD").split()  # whitespace-separated
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_snv.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_snv.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "var_attn")

cfg = importlib.import_module(f"model_config_{CONFIG_NAME}")
MODEL_NAME = ("TCGA_PanCan_SNV_baseline" if CONFIG_NAME == "baseline"
              else f"TCGA_PanCan_SNV_{CONFIG_NAME}_{CONTEXT_LEN}")

out_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
os.makedirs(out_dir, exist_ok=True)

# ============================================================================
# Load data and restrict to selected tumor types
# ============================================================================
train_data = pd.read_csv(TRAIN_DATA_PATH)
valid_data = pd.read_csv(VALID_DATA_PATH)
train_data = train_data[train_data["type"].isin(TUMOR_TYPES)]
valid_data = valid_data[valid_data["type"].isin(TUMOR_TYPES)]

# ============================================================================
# Build model and run inference
# ============================================================================
model = TESSERA(name=MODEL_NAME, use_distributed=cfg.use_distributed,
                jit_compile=False, mixed_precision=False)
ref_len = alt_len = 1 if cfg.mut_type == "SNV" else 10

model.create_sample_dataset(
    train_data["Tumor_Sample_Barcode"].values,
    train_data["Chromosome"].values,
    train_data["Start_Position"].values,
    train_data["Reference_Allele"].values,
    train_data["Tumor_Seq_Allele2"].values,
    vaf=train_data["vaf"].values, name="train_dataset",
    context_len=CONTEXT_LEN, batch_size=cfg.batch_size,
    is_training=False, subsample=cfg.subsample, fixed_bag_size=True,
    ref_len=ref_len, alt_len=alt_len,
)
model.create_sample_dataset(
    valid_data["Tumor_Sample_Barcode"].values,
    valid_data["Chromosome"].values,
    valid_data["Start_Position"].values,
    valid_data["Reference_Allele"].values,
    valid_data["Tumor_Seq_Allele2"].values,
    vaf=valid_data["vaf"].values, name="valid_dataset",
    context_len=CONTEXT_LEN, batch_size=cfg.batch_size,
    is_training=False, subsample=cfg.subsample, fixed_bag_size=True,
    ref_len=ref_len, alt_len=alt_len,
)

attn_train = model.get_global_attention_weights(dataset_name="train_dataset", agg="head")
attn_valid = model.get_global_attention_weights(dataset_name="valid_dataset", agg="head")
attn_df_train = create_attention_df(attn_train, train_data)
attn_df_valid = create_attention_df(attn_valid, valid_data)

out_path = os.path.join(out_dir, f"{'_'.join(TUMOR_TYPES)}.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"train": attn_df_train, "valid": attn_df_valid}, f)
print(f"Wrote {out_path}")
