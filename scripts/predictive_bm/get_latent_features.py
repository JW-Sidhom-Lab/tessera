"""
Extract SNV and CNA latent features from MSK-CHORD data using the trained
TCGA SNV+CNA model (InfoNCE without LOH).

Inputs (all produced by the data-creation scripts under
scripts/data/msk_chord/):
    ../data/msk_chord/snv.csv              (create_snv.py)
    ../data/msk_chord/cna_panel_filtered.csv
        OR ../data/msk_chord/cna.csv       (selected via CNA_DATA_SOURCE below)
    models/TCGA_SNV_CNA_InfoNCE_noLOH/best_model.keras   (produced by scripts/tcga_pancan_snv_cna/fit_model.py)

Output:
    msk_chord_latent_features_<CNA_DATA_SOURCE>.pkl - dict with:
        variant_features  - SNV embeddings
        cna_features      - CNA embeddings
        data_snv          - SNV metadata (same rows as variant_features)
        data_cna          - CNA metadata (same rows as cna_features)
"""

import pickle
import os
import json
import pandas as pd
import numpy as np

from tessera.model import TESSERA

CONTEXT_LEN = 25          # MUST match the context_len the model was trained with
                          # (tcga_pancan_snv_cna/model_config_infonce.py → context_len = 25)
BATCH_SIZE = 24

# Path to the trained multi-modal InfoNCE-noLOH model (best_model.keras lives here).
# Must be the same directory the model was saved to by fit_model.py.
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "../tcga_pancan_snv_cna/models/TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH",
)

# CNA input source.  Options:
#   'panel_filtered' - cna_panel_filtered.csv  (per-gene cross-joined, ~11.7M rows;
#                                                default, matches prior runs)
#   'raw'            - cna.csv                  (raw MSK-CHORD segments, ~1.3M rows;
#                                                coarser, closer to TCGA segmentation)
CNA_DATA_SOURCE = 'raw'
_CNA_PATHS = {
    'panel_filtered': '../data/msk_chord/cna_panel_filtered.csv',
    'raw'           : '../data/msk_chord/cna.csv',
}
if CNA_DATA_SOURCE not in _CNA_PATHS:
    raise ValueError(f"CNA_DATA_SOURCE must be one of {list(_CNA_PATHS)}, got {CNA_DATA_SOURCE!r}")
CNA_CSV_PATH = _CNA_PATHS[CNA_DATA_SOURCE]

# Output PKL is tagged with the CNA source so runs don't collide.
OUTPUT_FILE = f'msk_chord_latent_features_{CNA_DATA_SOURCE}.pkl'

# CNA normalization (match distribution of MSK-CHORD panel data to TCGA WES):
#   'none'     - pass raw Segment_Mean values through (baseline)
#   'linear'   - rescale to match TCGA global mean/std
#   'quantile' - rank-map onto TCGA Segment_Mean distribution (matches full shape)
NORMALIZATION_MODE    = 'quantile'
TCGA_CNA_STATS_PATH   = '../data/tcga/cna_stats.json'
TCGA_CNA_SORTED_PATH  = '../data/tcga/cna_sorted.npy'
TCGA_CNA_DATA_PATHS   = ['../data/tcga/train_data_cna.csv',
                         '../data/tcga/valid_data_cna.csv']


from tessera.data.preprocessing import (
    get_tcga_cna_stats,
    get_tcga_cna_sorted,
    quantile_normalize_to_tcga,
)

print("=" * 80)
print("Extracting Latent Features from MSK-CHORD Data")
print("=" * 80)

print("\n1. Loading SNV data...")
data_snv = pd.read_csv('../data/msk_chord/snv.csv')

print("   Calculating VAF from allele counts...")
data_snv['vaf'] = data_snv['t_alt_count'] / (data_snv['t_ref_count'] + data_snv['t_alt_count'])
data_snv['vaf'] = data_snv['vaf'].fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Total SNV records: {len(data_snv):,}")
print(f"   Unique samples: {data_snv['Tumor_Sample_Barcode'].nunique():,}")
print(f"   VAF range: {data_snv['vaf'].min():.4f} - {data_snv['vaf'].max():.4f}")

sample_ids = data_snv['Tumor_Sample_Barcode'].values
chromosomes = data_snv['Chromosome'].astype(str).values
positions = data_snv['Start_Position'].astype(int).values
refs = data_snv['Reference_Allele'].values
alts = data_snv['Tumor_Seq_Allele2'].values
vafs = data_snv['vaf'].values

print("\n2. Loading panel-filtered CNA data...")
print(f"   CNA data source: {CNA_DATA_SOURCE}  ({CNA_CSV_PATH})")
data_cna = pd.read_csv(CNA_CSV_PATH)

print(f"   Total CNA segments: {len(data_cna):,}")
print(f"   Unique samples with CNA: {data_cna['Tumor_Sample_Barcode'].nunique():,}")
print(f"   Segment Mean range: {data_cna['Segment_Mean'].min():.4f} - {data_cna['Segment_Mean'].max():.4f}")

cna_sample_ids = data_cna['Tumor_Sample_Barcode'].values
cna_chromosomes = data_cna['Chromosome'].astype(str).values
cna_starts = data_cna['Start'].astype(int).values
cna_ends = data_cna['End'].astype(int).values
cna_segment_means = data_cna['Segment_Mean'].astype(float).values

# Normalize MSK cna_segment_means to match TCGA distribution before inference
if NORMALIZATION_MODE != 'none':
    msk_mean_b, msk_std_b = float(cna_segment_means.mean()), float(cna_segment_means.std())
    msk_min_b, msk_max_b  = float(cna_segment_means.min()),  float(cna_segment_means.max())

    if NORMALIZATION_MODE == 'linear':
        tcga_mean, tcga_std = get_tcga_cna_stats(TCGA_CNA_STATS_PATH, TCGA_CNA_DATA_PATHS)
        print(f"\nLinear rescale MSK Segment_Mean to TCGA mean/std "
              f"(target mean={tcga_mean:.4f} std={tcga_std:.4f})")
        if msk_std_b == 0:
            raise ValueError("MSK Segment_Mean std is 0 - cannot rescale.")
        cna_segment_means = (cna_segment_means - msk_mean_b) / msk_std_b * tcga_std + tcga_mean

    elif NORMALIZATION_MODE == 'quantile':
        tcga_sorted = get_tcga_cna_sorted(TCGA_CNA_SORTED_PATH, TCGA_CNA_DATA_PATHS)
        print(f"\nQuantile normalize MSK Segment_Mean to TCGA distribution "
              f"(TCGA anchors n={tcga_sorted.size:,}, "
              f"range=[{tcga_sorted[0]:.3f}, {tcga_sorted[-1]:.3f}])")
        cna_segment_means = quantile_normalize_to_tcga(cna_segment_means, tcga_sorted).astype(np.float32)

    else:
        raise ValueError(f"Unknown NORMALIZATION_MODE={NORMALIZATION_MODE!r}")

    print(f"  MSK before:  mean={msk_mean_b:.4f} std={msk_std_b:.4f} "
          f"range=[{msk_min_b:.3f}, {msk_max_b:.3f}]")
    print(f"  MSK after :  mean={cna_segment_means.mean():.4f} "
          f"std={cna_segment_means.std():.4f} "
          f"range=[{cna_segment_means.min():.3f}, {cna_segment_means.max():.3f}]")

print("\n3. Initializing TESSERA model...")
print(f"   Model dir: {MODEL_DIR}")
model = TESSERA(
    model_dir=MODEL_DIR,
    use_distributed=False,
    jit_compile=False,
    mixed_precision=False,
)
print("   ✓ Model initialized")

ref_len = 1
alt_len = 1
print("\n4. Sequence configuration:")
print(f"   Reference length: {ref_len}")
print(f"   Alternate length: {alt_len}")
print(f"   Context length: {CONTEXT_LEN}")

print(f"\n5. Creating multi-modal dataset (is_training=False, batch_size={BATCH_SIZE})...")
model.create_sample_dataset(
    sample_ids=sample_ids,
    chromosomes=chromosomes,
    positions=positions,
    refs=refs,
    alts=alts,
    vaf=vafs,
    context_len=CONTEXT_LEN,
    batch_size=BATCH_SIZE,
    name='inference_dataset',
    is_training=False,
    subsample=None,
    fixed_bag_size=True,
    ref_len=ref_len,
    alt_len=alt_len,
    cna_sample_ids=cna_sample_ids,
    cna_chromosomes=cna_chromosomes,
    cna_starts=cna_starts,
    cna_ends=cna_ends,
    cna_segment_means=cna_segment_means,
    # MSK-CHORD panel CNA has no LOH annotations. Requires a TCGA model
    # trained with predict_cna_loh=False (fit_model.py now passes cna_lohs=None
    # when predict_cna_loh=False, so the model graph has no cna_loh input).
    # If you hit "Missing data for input 'cna_loh'", the loaded model was
    # trained with LOH as input - retrain with predict_cna_loh=False.
    cna_lohs=None,
    cna_subsample=None,
    z_score_cna=False,    # do NOT re-normalize - Segment_Mean is already quantile-matched to TCGA above
    z_score_clip=None,    # and the TCGA-trained model was itself trained on raw log2 ratios
)
print("   ✓ Dataset created")

print("\n6. Extracting SNV features...")
variant_features = model.get_variant_features(
    dataset_name='inference_dataset',
    downcast=False,
)
print(f"   ✓ SNV features shape: {variant_features.shape}")

print("\n7. Extracting CNA features...")
cna_features = model.get_cna_features(
    dataset_name='inference_dataset',
    downcast=False,
)
print(f"   ✓ CNA features shape: {cna_features.shape}")

print("\n8. Saving features...")
output_data = {
    'variant_features': variant_features,
    'cna_features': cna_features,
    'data_snv': data_snv,
    'data_cna': data_cna,
}
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(output_data, f)

file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
print(f"   ✓ Features saved to: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

print("\n" + "=" * 80)
print("✓ FEATURE EXTRACTION COMPLETE")
print("=" * 80)
print(f"  SNV features: {variant_features.shape}  ({len(data_snv):,} metadata rows)")
print(f"  CNA features: {cna_features.shape}  ({len(data_cna):,} metadata rows)")
print("=" * 80 + "\n")
