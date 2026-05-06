"""Extract per-segment predictions and reconstruction metrics from a trained TESSERA CNA model.

Loads a trained CNA model, runs it in inference mode on the train and
valid sets, and saves predicted vs actual segment-mean values plus
predicted vs actual LOH labels (when the model has the LOH head)
to ``cna_loss/<model_name>_cna_metrics.pkl``. Computes summary
metrics: MSE / MAE / R^2 / Pearson for segment-mean; accuracy /
AUC-ROC / precision / recall / F1 for LOH.

Used downstream by ``plot_cna_reconstruction.py`` (Figure 2 a-c) and
by the MSK-CHORD CNA cross-platform validation (Figure 2 d).

Inputs
------
``../data/tcga/train_data_cna.csv``
``../data/tcga/valid_data_cna.csv``

Output
------
``cna_loss/<model_name>_cna_metrics.pkl``

Usage
-----
    python get_cna_loss_metrics.py                   # attn_0 (default)
    CNA_ATTN_BLOCKS=2 python get_cna_loss_metrics.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score)

from tessera.model import TESSERA

import model_config

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
CNA_ATTN_BLOCKS = int(os.environ.get("CNA_ATTN_BLOCKS", "0"))
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

train_cna_loh = (train_data_cna["LOH"].astype(bool).values
                 if "LOH" in train_data_cna.columns else None)
valid_cna_loh = (valid_data_cna["LOH"].astype(bool).values
                 if "LOH" in valid_data_cna.columns else None)

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
    cna_lohs=train_cna_loh,
    name="train_dataset", **dataset_kwargs,
)
model.create_sample_dataset(
    cna_sample_ids=valid_data_cna["Tumor_Sample_Barcode"].values,
    cna_chromosomes=valid_data_cna["Chromosome"].values,
    cna_starts=valid_data_cna["Start"].astype(int).values,
    cna_ends=valid_data_cna["End"].astype(int).values,
    cna_segment_means=valid_data_cna["Segment_Mean"].astype(float).values,
    cna_lohs=valid_cna_loh,
    name="valid_dataset", **dataset_kwargs,
)

train_preds, train_actual, train_loh_preds, train_loh_actual = (
    model.get_cna_predictions("train_dataset", return_true_values=True, return_loh=True))
valid_preds, valid_actual, valid_loh_preds, valid_loh_actual = (
    model.get_cna_predictions("valid_dataset", return_true_values=True, return_loh=True))


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


def loh_metrics(preds: np.ndarray, actual: np.ndarray) -> dict:
    binary = (preds > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        actual, binary, average="binary", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, binary)),
        "auc_roc": float(roc_auc_score(actual, preds)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


train_metrics = segment_mean_metrics(train_preds, train_actual)
valid_metrics = segment_mean_metrics(valid_preds, valid_actual)
print(f"Train segment-mean: {train_metrics}")
print(f"Valid segment-mean: {valid_metrics}")

train_loh_m = (loh_metrics(train_loh_preds, train_loh_actual)
               if train_loh_preds is not None and train_loh_actual is not None
               else None)
valid_loh_m = (loh_metrics(valid_loh_preds, valid_loh_actual)
               if valid_loh_preds is not None and valid_loh_actual is not None
               else None)
if train_loh_m is not None:
    print(f"Train LOH:          {train_loh_m}")
    print(f"Valid LOH:          {valid_loh_m}")

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
        "train_loh_predictions": train_loh_preds,
        "train_loh_actual": train_loh_actual,
        "valid_loh_predictions": valid_loh_preds,
        "valid_loh_actual": valid_loh_actual,
        "train_data_cna": train_data_cna,
        "valid_data_cna": valid_data_cna,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "train_loh_metrics": train_loh_m,
        "valid_loh_metrics": valid_loh_m,
        "model_name": MODEL_NAME,
        "cna_attention_blocks": CNA_ATTN_BLOCKS,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote {out_path}")
