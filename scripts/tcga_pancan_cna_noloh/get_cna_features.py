"""Extract per-segment latent features from a trained TESSERA CNA NoLOH model.

Inputs
------
``../data/tcga/train_data_cna.csv``
``../data/tcga/valid_data_cna.csv``

Output
------
``cna_features/<model_name>_cna_features.pkl``

Usage
-----
    python get_cna_features.py                   # attn_2 (default)
    CNA_ATTN_BLOCKS=1 python get_cna_features.py
"""

import os
import pickle

import pandas as pd

from tessera.model import TESSERA

import model_config

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
CNA_ATTN_BLOCKS = int(os.environ.get("CNA_ATTN_BLOCKS", "2"))
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_cna.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_cna.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "cna_features")

MODEL_NAME = f"{model_config.model_name}_attn_{CNA_ATTN_BLOCKS}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load data
# ============================================================================
train_data_cna = pd.read_csv(TRAIN_DATA_PATH)
valid_data_cna = pd.read_csv(VALID_DATA_PATH)

# ============================================================================
# Build model and run inference
# ============================================================================
model = TESSERA(name=MODEL_NAME, use_distributed=model_config.use_distributed,
                jit_compile=False, mixed_precision=False)

dataset_kwargs = dict(
    cna_subsample=model_config.cna_subsample,
    batch_size=model_config.batch_size,
    is_training=False, fixed_bag_size=True,
    z_score_cna=model_config.z_score_cna,
    z_score_clip=model_config.z_score_clip,
)
model.create_sample_dataset(
    cna_sample_ids=train_data_cna["Tumor_Sample_Barcode"].values,
    cna_chromosomes=train_data_cna["Chromosome"].values,
    cna_starts=train_data_cna["Start"].astype(int).values,
    cna_ends=train_data_cna["End"].astype(int).values,
    cna_segment_means=train_data_cna["Segment_Mean"].astype(float).values,
    cna_lohs=None,
    name="train_dataset", **dataset_kwargs,
)
model.create_sample_dataset(
    cna_sample_ids=valid_data_cna["Tumor_Sample_Barcode"].values,
    cna_chromosomes=valid_data_cna["Chromosome"].values,
    cna_starts=valid_data_cna["Start"].astype(int).values,
    cna_ends=valid_data_cna["End"].astype(int).values,
    cna_segment_means=valid_data_cna["Segment_Mean"].astype(float).values,
    cna_lohs=None,
    name="valid_dataset", **dataset_kwargs,
)

train_features_cna = model.get_cna_features("train_dataset", downcast=False)
valid_features_cna = model.get_cna_features("valid_dataset", downcast=False)

out_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_cna_features.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "train_features_cna": train_features_cna,
        "valid_features_cna": valid_features_cna,
        "train_data_cna": train_data_cna,
        "valid_data_cna": valid_data_cna,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote {out_path}")
