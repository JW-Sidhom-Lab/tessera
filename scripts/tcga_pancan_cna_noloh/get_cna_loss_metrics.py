"""Extract per-segment predictions and segment-mean reconstruction metrics from a trained TESSERA CNA NoLOH model.

Loads a trained NoLOH CNA model, runs it in inference mode on the train
and valid sets, and saves predicted vs actual segment-mean values plus
summary metrics (MSE / MAE / R^2 / Pearson) to
``cna_loss/<model_name>_cna_metrics.pkl``. No LOH outputs (the NoLOH
variant has no LOH head).

Inputs
------
``../data/tcga/train_data_cna.csv``
``../data/tcga/valid_data_cna.csv``

Output
------
``cna_loss/<model_name>_cna_metrics.pkl``

Usage
-----
    python get_cna_loss_metrics.py                   # attn_2 (default)
    CNA_ATTN_BLOCKS=1 python get_cna_loss_metrics.py
"""

import os
import pickle

import numpy as np
import pandas as pd

from tessera.model import TESSERA

import model_config

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
CNA_ATTN_BLOCKS = int(os.environ.get("CNA_ATTN_BLOCKS", "2"))
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_cna.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_cna.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "cna_loss")

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

train_preds, train_actual = model.get_cna_predictions("train_dataset", return_true_values=True)
valid_preds, valid_actual = model.get_cna_predictions("valid_dataset", return_true_values=True)


# ============================================================================
# Metrics
# ============================================================================

def segment_mean_metrics(preds: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "mse": float(np.mean((preds - actual) ** 2)),
        "mae": float(np.mean(np.abs(preds - actual))),
        "r2": float(1 - (np.sum((actual - preds) ** 2)
                         / np.sum((actual - np.mean(actual)) ** 2))),
        "correlation": float(np.corrcoef(actual, preds)[0, 1]),
    }


train_metrics = segment_mean_metrics(train_preds, train_actual)
valid_metrics = segment_mean_metrics(valid_preds, valid_actual)
print(f"Train segment-mean: {train_metrics}")
print(f"Valid segment-mean: {valid_metrics}")

# ============================================================================
# Save
# ============================================================================
out_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_cna_metrics.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "train_predictions": train_preds,
        "train_actual": train_actual,
        "valid_predictions": valid_preds,
        "valid_actual": valid_actual,
        "train_data_cna": train_data_cna,
        "valid_data_cna": valid_data_cna,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "model_name": MODEL_NAME,
        "cna_attention_blocks": CNA_ATTN_BLOCKS,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote {out_path}")
