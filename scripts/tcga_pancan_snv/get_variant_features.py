"""Extract per-variant latent features from a trained TESSERA SNV model.

Loads the TCGA training and validation SNV tables plus a trained model
(produced by ``fit_model.py``), runs the model in inference mode, and
saves per-variant latent features to ``var_features/<model_name>_features.pkl``.
The output is a dict with ``train`` and ``valid`` arrays.

Inputs
------
``../data/tcga/train_data_snv.csv``
``../data/tcga/valid_data_snv.csv``

Output
------
``var_features/<model_name>_features.pkl``

Usage
-----
    python get_variant_features.py                                   # baseline / 1
    CONFIG=local CONTEXT_LEN=25 python get_variant_features.py
"""

import importlib
import os
import pickle

import pandas as pd

from tessera.model import TESSERA

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
CONFIG_NAME = os.environ.get("CONFIG", "baseline")          # baseline | local | global
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "1"))
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_snv.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_snv.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "var_features")

cfg = importlib.import_module(f"model_config_{CONFIG_NAME}")
MODEL_NAME = ("TCGA_PanCan_SNV_baseline" if CONFIG_NAME == "baseline"
              else f"TCGA_PanCan_SNV_{CONFIG_NAME}_{CONTEXT_LEN}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load data
# ============================================================================
train_data = pd.read_csv(TRAIN_DATA_PATH)
valid_data = pd.read_csv(VALID_DATA_PATH)

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

train_features = model.get_variant_features(dataset_name="train_dataset", downcast=False)
valid_features = model.get_variant_features(dataset_name="valid_dataset", downcast=False)

out_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_features.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"train": train_features, "valid": valid_features}, f)
print(f"Wrote {out_path}")
