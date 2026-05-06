"""Per-token loss + CNA reconstruction metrics from the trained joint model.

Loads the trained joint SNV+CNA model produced by ``fit_model.py`` and
computes:

- per-variant SNV reconstruction loss + logits + true alleles
- per-segment CNA Segment_Mean predictions + MSE / MAE / R^2 / Pearson
- (when LOH was trained) per-segment LOH probabilities + accuracy /
  AUC-ROC / precision / recall / F1

Inputs
------
``../data/tcga/{train,valid}_data_snv.csv``
``../data/tcga/{train,valid}_data_cna.csv``
Trained model in ``models/<model_name>/``.

Output
------
``multimodal_loss/<model_name>_multimodal_metrics.pkl``
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

from tessera.model import TESSERA

import model_config_infonce as model_config

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
TRAIN_SNV_PATH = os.environ.get("TRAIN_SNV", "../data/tcga/train_data_snv.csv")
VALID_SNV_PATH = os.environ.get("VALID_SNV", "../data/tcga/valid_data_snv.csv")
TRAIN_CNA_PATH = os.environ.get("TRAIN_CNA", "../data/tcga/train_data_cna.csv")
VALID_CNA_PATH = os.environ.get("VALID_CNA", "../data/tcga/valid_data_cna.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "multimodal_loss")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load data
# ============================================================================
print(f"Loading SNV: {TRAIN_SNV_PATH}, {VALID_SNV_PATH}")
train_data_snv = pd.read_csv(TRAIN_SNV_PATH)
valid_data_snv = pd.read_csv(VALID_SNV_PATH)

print(f"Loading CNA: {TRAIN_CNA_PATH}, {VALID_CNA_PATH}")
train_data_cna = pd.read_csv(TRAIN_CNA_PATH)
valid_data_cna = pd.read_csv(VALID_CNA_PATH)

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

train_cna_sample_ids = train_data_cna["Tumor_Sample_Barcode"].values
train_cna_chr = train_data_cna["Chromosome"].astype(str).values
train_cna_start = train_data_cna["Start"].astype(int).values
train_cna_end = train_data_cna["End"].astype(int).values
train_cna_segment_mean = train_data_cna["Segment_Mean"].astype(float).values
train_cna_loh = (train_data_cna["LOH"].astype(bool).values
                 if "LOH" in train_data_cna.columns else None)

valid_cna_sample_ids = valid_data_cna["Tumor_Sample_Barcode"].values
valid_cna_chr = valid_data_cna["Chromosome"].astype(str).values
valid_cna_start = valid_data_cna["Start"].astype(int).values
valid_cna_end = valid_data_cna["End"].astype(int).values
valid_cna_segment_mean = valid_data_cna["Segment_Mean"].astype(float).values
valid_cna_loh = (valid_data_cna["LOH"].astype(bool).values
                 if "LOH" in valid_data_cna.columns else None)

# ============================================================================
# Build inference-mode datasets
# ============================================================================
model = TESSERA(name=model_config.model_name,
                use_distributed=model_config.use_distributed,
                jit_compile=False, mixed_precision=False)

ref_len = alt_len = 1 if model_config.mut_type == "SNV" else 10

dataset_kwargs_common = dict(
    context_len=model_config.context_len,
    batch_size=model_config.batch_size,
    is_training=False,
    subsample=model_config.subsample,
    fixed_bag_size=True,
    ref_len=ref_len, alt_len=alt_len,
    cna_subsample=model_config.cna_subsample,
)

model.create_sample_dataset(
    sample_ids=train_sample_ids, chromosomes=train_chr, positions=train_pos,
    refs=train_ref, alts=train_alt, vaf=train_vaf,
    cna_sample_ids=train_cna_sample_ids, cna_chromosomes=train_cna_chr,
    cna_starts=train_cna_start, cna_ends=train_cna_end,
    cna_segment_means=train_cna_segment_mean, cna_lohs=train_cna_loh,
    name="train_dataset", **dataset_kwargs_common,
)
model.create_sample_dataset(
    sample_ids=valid_sample_ids, chromosomes=valid_chr, positions=valid_pos,
    refs=valid_ref, alts=valid_alt, vaf=valid_vaf,
    cna_sample_ids=valid_cna_sample_ids, cna_chromosomes=valid_cna_chr,
    cna_starts=valid_cna_start, cna_ends=valid_cna_end,
    cna_segment_means=valid_cna_segment_mean, cna_lohs=valid_cna_loh,
    name="valid_dataset", **dataset_kwargs_common,
)

# ============================================================================
# SNV per-variant predictions + loss
# ============================================================================
print("Computing SNV predictions...")
out_train_snv = model.get_variant_probabilities(
    dataset_name="train_dataset",
    return_logits=True, return_true_values=True, return_loss=True,
    non_zero_only=False, return_ref=True,
)
out_valid_snv = model.get_variant_probabilities(
    dataset_name="valid_dataset",
    return_logits=True, return_true_values=True, return_loss=True,
    non_zero_only=False, return_ref=True,
)

# ============================================================================
# CNA Segment_Mean + LOH predictions and metrics
# ============================================================================
print("Computing CNA predictions...")
train_cna_preds, train_cna_actual, train_loh_preds, train_loh_actual = \
    model.get_cna_predictions("train_dataset", return_true_values=True, return_loh=True)
valid_cna_preds, valid_cna_actual, valid_loh_preds, valid_loh_actual = \
    model.get_cna_predictions("valid_dataset", return_true_values=True, return_loh=True)


def regression_metrics(preds, actual):
    mse = np.mean((preds - actual) ** 2)
    mae = np.mean(np.abs(preds - actual))
    r2 = 1 - np.sum((actual - preds) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    corr = np.corrcoef(actual, preds)[0, 1]
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2), "correlation": float(corr)}


def classification_metrics(preds, actual):
    binary = (preds > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        actual, binary, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(actual, binary)),
        "auc_roc": float(roc_auc_score(actual, preds)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


cna_train_metrics = regression_metrics(train_cna_preds, train_cna_actual)
cna_valid_metrics = regression_metrics(valid_cna_preds, valid_cna_actual)

loh_train_metrics = (classification_metrics(train_loh_preds, train_loh_actual)
                     if train_loh_preds is not None and train_loh_actual is not None else None)
loh_valid_metrics = (classification_metrics(valid_loh_preds, valid_loh_actual)
                     if valid_loh_preds is not None and valid_loh_actual is not None else None)

print(f"CNA Segment_Mean valid R^2 = {cna_valid_metrics['r2']:.4f}")
if loh_valid_metrics is not None:
    print(f"CNA LOH valid AUC-ROC  = {loh_valid_metrics['auc_roc']:.4f}")

# ============================================================================
# Save
# ============================================================================
output_path = os.path.join(OUTPUT_DIR, f"{model_config.model_name}_multimodal_metrics.pkl")
with open(output_path, "wb") as f:
    pickle.dump({
        "snv_train": out_train_snv,
        "snv_valid": out_valid_snv,
        "cna_train_predictions": train_cna_preds,
        "cna_train_actual": train_cna_actual,
        "cna_valid_predictions": valid_cna_preds,
        "cna_valid_actual": valid_cna_actual,
        "cna_train_loh_predictions": train_loh_preds,
        "cna_train_loh_actual": train_loh_actual,
        "cna_valid_loh_predictions": valid_loh_preds,
        "cna_valid_loh_actual": valid_loh_actual,
        "train_data_snv": train_data_snv,
        "valid_data_snv": valid_data_snv,
        "train_data_cna": train_data_cna,
        "valid_data_cna": valid_data_cna,
        "cna_train_metrics": cna_train_metrics,
        "cna_valid_metrics": cna_valid_metrics,
        "loh_train_metrics": loh_train_metrics,
        "loh_valid_metrics": loh_valid_metrics,
        "model_name": model_config.model_name,
        "config": {
            "mut_type": model_config.mut_type,
            "context_len": model_config.context_len,
            "batch_size": model_config.batch_size,
            "use_cna": model_config.use_cna,
            "predict_cna_loh": model_config.predict_cna_loh,
            "cross_modal_blocks": model_config.cross_modal_blocks,
        },
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
