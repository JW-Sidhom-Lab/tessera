"""Extract per-variant + per-segment features from the trained joint model.

Loads the trained joint SNV+CNA model produced by ``fit_model.py``, runs
the SNV and CNA encoders in inference mode against the TCGA training and
validation tables, and saves both per-token (variant / CNA segment) and
per-sample-aligned latent features. Downstream:
``scripts/tumor_classification_snv_cna_infonce/`` consumes the resulting
pkl for the InfoNCE-aligned tumour-type classifier.

Inputs
------
``../data/tcga/{train,valid}_data_snv.csv``
``../data/tcga/{train,valid}_data_cna.csv``
Trained model in ``models/<model_name>/``.

Output
------
``multimodal_features/<model_name>_multimodal_features.pkl``
"""

import os
import pickle

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
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "multimodal_features")

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

# CNA columns. Same gating on LOH as during training.
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
# Build inference-mode datasets and run encoders
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
    z_score_cna=model_config.z_score_cna,
    z_score_clip=model_config.z_score_clip,
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

print("Extracting variant features...")
train_variant_features = model.get_variant_features(dataset_name="train_dataset", downcast=False)
valid_variant_features = model.get_variant_features(dataset_name="valid_dataset", downcast=False)
print(f"  train: {train_variant_features.shape}, valid: {valid_variant_features.shape}")

print("Extracting CNA features...")
train_cna_features = model.get_cna_features(dataset_name="train_dataset", downcast=False)
valid_cna_features = model.get_cna_features(dataset_name="valid_dataset", downcast=False)
print(f"  train: {train_cna_features.shape}, valid: {valid_cna_features.shape}")

# ============================================================================
# Save
# ============================================================================
output_path = os.path.join(OUTPUT_DIR, f"{model_config.model_name}_multimodal_features.pkl")
with open(output_path, "wb") as f:
    pickle.dump({
        "train_variant_features": train_variant_features,
        "valid_variant_features": valid_variant_features,
        "train_cna_features": train_cna_features,
        "valid_cna_features": valid_cna_features,
        "train_data_snv": train_data_snv,
        "valid_data_snv": valid_data_snv,
        "train_data_cna": train_data_cna,
        "valid_data_cna": valid_data_cna,
    }, f)
print(f"Wrote {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
