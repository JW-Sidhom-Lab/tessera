"""Extract SNV and CNA latent features from DepMap 24Q2 cell lines.

Runs the TCGA-pretrained TESSERA SNV+CNA InfoNCE-noLOH model on the
MSK-IMPACT505 panel-restricted DepMap inputs produced by
``scripts/data/depmap/prepare_depmap_data.py``. Panel-restriction matches
the assay coverage that MSK-CHORD's beta_eff was fitted on; the attention
weights depend on bag composition, so feeding full WES would produce
embeddings that do not transfer.

Inputs:
    ../data/depmap/snv_panel.csv         (TCGA-format hg19, panel-restricted)
    ../data/depmap/cna_panel.csv         (log2-ratio segments, hg19, panel-restricted)
    ../tcga_pancan_snv_cna/models/TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH/best_model.keras

Output:
    depmap_latent_features_panel.pkl, a dict with:
        variant_features    SNV embeddings
        cna_features        CNA embeddings
        data_snv            SNV metadata (same row count as variant_features)
        data_cna            CNA metadata (same row count as cna_features)

Run on RunPod (Keras model inference is too slow on Mac).
"""

import pickle
import os
import json
import pandas as pd
import numpy as np

from tessera.model import TESSERA


# Must match what the model was trained with
# (tcga_pancan_snv_cna/model_config_infonce.py: context_len = 25)
CONTEXT_LEN = 25
BATCH_SIZE = 24

# Path to the trained multi-modal InfoNCE-noLOH model (same as MSK-CHORD).
MODEL_DIR = '../tcga_pancan_snv_cna/models/TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH'

# DepMap MSK-IMPACT505 panel-restricted inputs.
SNV_CSV_PATH = '../data/depmap/snv_panel.csv'
CNA_CSV_PATH = '../data/depmap/cna_panel.csv'
OUTPUT_FILE  = 'depmap_latent_features_panel.pkl'

# CNA normalization (match DepMap log2-ratio distribution to TCGA WES):
#   'none'     - pass raw Segment_Mean values through (baseline)
#   'linear'   - rescale to match TCGA global mean/std
#   'quantile' - rank-map onto TCGA Segment_Mean distribution (matches full shape)
#
# Cell lines have systematically higher ploidy / more aneuploidy than primary
# tumours; without normalization the log2-ratio distribution is shifted vs
# TCGA. Quantile-normalisation (the same method used for MSK-IMPACT panel
# segments) handles this without retraining the model.
NORMALIZATION_MODE    = 'quantile'
TCGA_CNA_STATS_PATH   = '../data/tcga/cna_stats.json'
TCGA_CNA_SORTED_PATH  = '../data/tcga/cna_sorted.npy'
TCGA_CNA_DATA_PATHS   = ['../data/tcga/train_data_cna.csv',
                         '../data/tcga/valid_data_cna.csv']


def _load_tcga_segment_mean(data_paths):
    vals = []
    for p in data_paths:
        df = pd.read_csv(p, usecols=['Segment_Mean'])
        vals.append(df['Segment_Mean'].astype(float).values)
    return np.concatenate(vals)


def get_tcga_cna_stats(stats_path, data_paths):
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            s = json.load(f)
        return float(s['mean']), float(s['std'])
    print(f"Computing TCGA CNA stats (cache miss: {stats_path})")
    arr = _load_tcga_segment_mean(data_paths)
    mean, std = float(arr.mean()), float(arr.std())
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump({'mean': mean, 'std': std, 'n_segments': int(arr.size)}, f, indent=2)
    print(f"  Cached to {stats_path}: mean={mean:.4f} std={std:.4f} (n={arr.size:,})")
    return mean, std


def get_tcga_cna_sorted(sorted_path, data_paths):
    if os.path.exists(sorted_path):
        return np.load(sorted_path)
    print(f"Computing sorted TCGA CNA array (cache miss: {sorted_path})")
    arr = np.sort(_load_tcga_segment_mean(data_paths)).astype(np.float32)
    os.makedirs(os.path.dirname(sorted_path), exist_ok=True)
    np.save(sorted_path, arr)
    print(f"  Cached to {sorted_path}: n={arr.size:,}")
    return arr


def quantile_normalize_to_tcga(vals, tcga_sorted):
    from scipy.stats import rankdata
    n = len(vals)
    ranks = rankdata(vals, method='average')
    q     = (ranks - 0.5) / n
    tcga_q = np.linspace(0.0, 1.0, len(tcga_sorted))
    return np.interp(q, tcga_q, tcga_sorted)


print("=" * 80)
print("Extracting Latent Features from DepMap 24Q2 Cell-Line Data")
print("=" * 80)

print("\n1. Loading SNV data...")
data_snv = pd.read_csv(SNV_CSV_PATH)

# Match MSK-CHORD inference: filter to single-base substitutions (the model
# was trained with ref_len=alt_len=1; indels would crash the SNV encoder).
# DepMap reports 'SNV' for substitutions, 'insertion' / 'deletion' otherwise.
n_before = len(data_snv)
data_snv = data_snv[data_snv['VariantType'] == 'SNV'].copy()
# Belt-and-braces: also drop any rows where ref or alt isn't a single base
data_snv = data_snv[
    (data_snv['Reference_Allele'].str.len() == 1)
    & (data_snv['Tumor_Seq_Allele2'].str.len() == 1)
    & (data_snv['Reference_Allele'] != '-')
    & (data_snv['Tumor_Seq_Allele2'] != '-')
].copy()
print(f"   Filtered indels: {n_before:,} → {len(data_snv):,} SNV-only rows "
      f"({n_before - len(data_snv):,} dropped)")

# DepMap prep saves both 'vaf' (precomputed) and t_ref/t_alt counts. Recompute
# from counts to match MSK-CHORD inference convention exactly.
print("   Calculating VAF from allele counts...")
data_snv['vaf'] = data_snv['t_alt_count'] / (
    data_snv['t_ref_count'] + data_snv['t_alt_count']
)
data_snv['vaf'] = data_snv['vaf'].fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Total SNV records: {len(data_snv):,}")
print(f"   Unique cell lines: {data_snv['Tumor_Sample_Barcode'].nunique():,}")
print(f"   VAF range: {data_snv['vaf'].min():.4f} - {data_snv['vaf'].max():.4f}")

sample_ids   = data_snv['Tumor_Sample_Barcode'].values
chromosomes  = data_snv['Chromosome'].astype(str).values
positions    = data_snv['Start_Position'].astype(int).values
refs         = data_snv['Reference_Allele'].values
alts         = data_snv['Tumor_Seq_Allele2'].values
vafs         = data_snv['vaf'].values

print("\n2. Loading CNA segment data...")
print(f"   CNA file: {CNA_CSV_PATH}")
data_cna = pd.read_csv(CNA_CSV_PATH)

print(f"   Total CNA segments: {len(data_cna):,}")
print(f"   Unique cell lines with CNA: {data_cna['Tumor_Sample_Barcode'].nunique():,}")
print(f"   Segment Mean range: {data_cna['Segment_Mean'].min():.4f} - "
      f"{data_cna['Segment_Mean'].max():.4f}")

cna_sample_ids    = data_cna['Tumor_Sample_Barcode'].values
cna_chromosomes   = data_cna['Chromosome'].astype(str).values
cna_starts        = data_cna['Start'].astype(int).values
cna_ends          = data_cna['End'].astype(int).values
cna_segment_means = data_cna['Segment_Mean'].astype(float).values

# Normalize DepMap Segment_Mean to TCGA distribution before inference. Cell
# lines are aneuploid / hyperdiploid relative to primary tumours; the log2-
# ratio distribution is right-shifted, so without normalisation the encoder
# sees out-of-distribution input. Same approach used for MSK-IMPACT segments.
if NORMALIZATION_MODE != 'none':
    depmap_mean_b = float(cna_segment_means.mean())
    depmap_std_b  = float(cna_segment_means.std())
    depmap_min_b  = float(cna_segment_means.min())
    depmap_max_b  = float(cna_segment_means.max())

    if NORMALIZATION_MODE == 'linear':
        tcga_mean, tcga_std = get_tcga_cna_stats(TCGA_CNA_STATS_PATH,
                                                  TCGA_CNA_DATA_PATHS)
        print(f"\nLinear rescale DepMap Segment_Mean to TCGA mean/std "
              f"(target mean={tcga_mean:.4f} std={tcga_std:.4f})")
        if depmap_std_b == 0:
            raise ValueError("DepMap Segment_Mean std is 0 - cannot rescale.")
        cna_segment_means = (cna_segment_means - depmap_mean_b) / depmap_std_b * tcga_std + tcga_mean

    elif NORMALIZATION_MODE == 'quantile':
        tcga_sorted = get_tcga_cna_sorted(TCGA_CNA_SORTED_PATH,
                                            TCGA_CNA_DATA_PATHS)
        print(f"\nQuantile normalize DepMap Segment_Mean to TCGA distribution "
              f"(TCGA anchors n={tcga_sorted.size:,}, "
              f"range=[{tcga_sorted[0]:.3f}, {tcga_sorted[-1]:.3f}])")
        cna_segment_means = quantile_normalize_to_tcga(
            cna_segment_means, tcga_sorted
        ).astype(np.float32)

    else:
        raise ValueError(f"Unknown NORMALIZATION_MODE={NORMALIZATION_MODE!r}")

    print(f"  DepMap before:  mean={depmap_mean_b:.4f} std={depmap_std_b:.4f} "
          f"range=[{depmap_min_b:.3f}, {depmap_max_b:.3f}]")
    print(f"  DepMap after :  mean={cna_segment_means.mean():.4f} "
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
    # DepMap CNA has LoH calls but the model graph here has no cna_loh input
    # (matches MSK-CHORD inference, which uses the noLOH-trained model variant).
    cna_lohs=None,
    cna_subsample=None,
    z_score_cna=False,    # already quantile-matched to TCGA above
    z_score_clip=None,
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
