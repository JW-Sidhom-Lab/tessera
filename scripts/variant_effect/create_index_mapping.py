"""Map dedup-split rows back to their feature-file positions in the original patient-level split.

The TESSERA per-variant features in
``../tcga_pancan_snv/var_features/*_features.pkl`` were computed for the
original patient-level row order from
``../data/tcga/{train,valid}_data_snv.csv``. After
``create_variant_dedup_split.py`` re-shuffles those rows into the
deduplicated split, we still want to fetch the right features without
re-running inference. This script writes
``data/index_mapping{,_gene}.pkl`` with arrays mapping each dedup row to
its position in the combined (train+valid) original feature space.

Inputs
------
``data/{train,valid}_data_snv_dedup{,_gene}.csv`` (from create_variant_dedup_split.py)
``../data/tcga/{train,valid}_data_snv.csv`` (original patient-level split)

Outputs
-------
``data/index_mapping{,_gene}.pkl``

Usage
-----
    python create_index_mapping.py                       # both strategies (default)
    STRATEGY=variant python create_index_mapping.py
    STRATEGY=gene    python create_index_mapping.py
"""

import os
import pickle

import numpy as np
import pandas as pd

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
STRATEGY = os.environ.get("STRATEGY", "both")
ORIGINAL_TRAIN_FILE = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_snv.csv")
ORIGINAL_VALID_FILE = os.environ.get("VALID_DATA", "../data/tcga/valid_data_snv.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")

def create_index_mapping(
    strategy='variant',
    dedup_train_file=None,
    dedup_valid_file=None,
    original_train_file='../data/tcga/train_data_snv.csv',
    original_valid_file='../data/tcga/valid_data_snv.csv',
    output_file=None,
    output_dir='data'
):
    """
    Create mapping from deduplicated row indices to original row indices.

    Parameters:
    -----------
    strategy : str
        Deduplication strategy: 'variant' or 'gene' (default: 'variant')
    dedup_train_file : str
        Path to deduplicated train CSV (auto-determined if None)
    dedup_valid_file : str
        Path to deduplicated valid CSV (auto-determined if None)
    original_train_file : str
        Path to original train CSV
    original_valid_file : str
        Path to original valid CSV
    output_file : str
        Path to save mapping pickle file (auto-determined if None)
    output_dir : str
        Directory for output files if auto-determining paths

    Returns:
    --------
    dict : Mapping with train and valid indices
    """

    # Determine file paths based on strategy
    if dedup_train_file is None or dedup_valid_file is None or output_file is None:
        if strategy == 'gene':
            dedup_train_file = dedup_train_file or os.path.join(output_dir, 'train_data_snv_dedup_gene.csv')
            dedup_valid_file = dedup_valid_file or os.path.join(output_dir, 'valid_data_snv_dedup_gene.csv')
            output_file = output_file or os.path.join(output_dir, 'index_mapping_gene.pkl')
        else:  # variant
            dedup_train_file = dedup_train_file or os.path.join(output_dir, 'train_data_snv_dedup.csv')
            dedup_valid_file = dedup_valid_file or os.path.join(output_dir, 'valid_data_snv_dedup.csv')
            output_file = output_file or os.path.join(output_dir, 'index_mapping.pkl')

    print("=" * 80)
    strategy_desc = "VARIANT-BASED" if strategy == 'variant' else "GENE-BASED"
    print(f"CREATING INDEX MAPPING FOR {strategy_desc} DEDUPLICATED DATA")
    print("=" * 80)

    # Load deduplicated data
    print("\n1. Loading deduplicated data...")
    dedup_train = pd.read_csv(dedup_train_file, usecols=['Tumor_Sample_Barcode', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])
    dedup_valid = pd.read_csv(dedup_valid_file, usecols=['Tumor_Sample_Barcode', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])

    print(f"   Dedup train rows: {len(dedup_train):,}")
    print(f"   Dedup valid rows: {len(dedup_valid):,}")

    # Load original data
    print("\n2. Loading original data...")
    orig_train = pd.read_csv(original_train_file, usecols=['Tumor_Sample_Barcode', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])
    orig_valid = pd.read_csv(original_valid_file, usecols=['Tumor_Sample_Barcode', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])

    print(f"   Original train rows: {len(orig_train):,}")
    print(f"   Original valid rows: {len(orig_valid):,}")

    # Create variant IDs for matching
    print("\n3. Creating variant IDs for matching...")

    dedup_train['var_id'] = (dedup_train['Chromosome'].astype(str) + '_' +
                            dedup_train['Start_Position'].astype(str) + '_' +
                            dedup_train['Reference_Allele'].astype(str) + '_' +
                            dedup_train['Tumor_Seq_Allele2'].astype(str))
    dedup_train['row_sample'] = dedup_train['Tumor_Sample_Barcode'] + '_' + dedup_train['var_id']

    dedup_valid['var_id'] = (dedup_valid['Chromosome'].astype(str) + '_' +
                            dedup_valid['Start_Position'].astype(str) + '_' +
                            dedup_valid['Reference_Allele'].astype(str) + '_' +
                            dedup_valid['Tumor_Seq_Allele2'].astype(str))
    dedup_valid['row_sample'] = dedup_valid['Tumor_Sample_Barcode'] + '_' + dedup_valid['var_id']

    orig_train['var_id'] = (orig_train['Chromosome'].astype(str) + '_' +
                           orig_train['Start_Position'].astype(str) + '_' +
                           orig_train['Reference_Allele'].astype(str) + '_' +
                           orig_train['Tumor_Seq_Allele2'].astype(str))
    orig_train['row_sample'] = orig_train['Tumor_Sample_Barcode'] + '_' + orig_train['var_id']

    orig_valid['var_id'] = (orig_valid['Chromosome'].astype(str) + '_' +
                           orig_valid['Start_Position'].astype(str) + '_' +
                           orig_valid['Reference_Allele'].astype(str) + '_' +
                           orig_valid['Tumor_Seq_Allele2'].astype(str))
    orig_valid['row_sample'] = orig_valid['Tumor_Sample_Barcode'] + '_' + orig_valid['var_id']

    print(f"   Dedup train unique row_samples: {dedup_train['row_sample'].nunique():,}")
    print(f"   Original train unique row_samples: {orig_train['row_sample'].nunique():,}")

    # Create mapping for train
    print("\n4. Creating train set mapping...")

    # Use a different approach: match by variant + sample across original combined data
    orig_combined = pd.concat([orig_train, orig_valid], ignore_index=True)
    orig_combined['var_id'] = (orig_combined['Chromosome'].astype(str) + '_' +
                               orig_combined['Start_Position'].astype(str) + '_' +
                               orig_combined['Reference_Allele'].astype(str) + '_' +
                               orig_combined['Tumor_Seq_Allele2'].astype(str))
    orig_combined['row_sample'] = orig_combined['Tumor_Sample_Barcode'] + '_' + orig_combined['var_id']
    orig_idx_map = {row_sample: idx for idx, row_sample in enumerate(orig_combined['row_sample'])}

    train_indices = []
    unmapped_train = 0

    for dedup_idx, row_sample in enumerate(dedup_train['row_sample']):
        if row_sample in orig_idx_map:
            train_indices.append(orig_idx_map[row_sample])
        else:
            print(f"   WARNING: Row {dedup_idx} not found in original data: {row_sample}")
            unmapped_train += 1
            # Use a placeholder or skip
            train_indices.append(-1)  # Mark as unmapped

    train_indices = np.array(train_indices, dtype=np.int32)

    if unmapped_train > 0:
        print(f"   ⚠ Unmapped train rows: {unmapped_train}")
    else:
        print(f"   ✓ All {len(train_indices):,} train rows mapped successfully")

    # Create mapping for valid
    print("\n5. Creating valid set mapping...")

    valid_indices = []
    unmapped_valid = 0

    for dedup_idx, row_sample in enumerate(dedup_valid['row_sample']):
        if row_sample in orig_idx_map:
            valid_indices.append(orig_idx_map[row_sample])
        else:
            print(f"   WARNING: Row {dedup_idx} not found in original data: {row_sample}")
            unmapped_valid += 1
            valid_indices.append(-1)  # Mark as unmapped

    valid_indices = np.array(valid_indices, dtype=np.int32)

    if unmapped_valid > 0:
        print(f"   ⚠ Unmapped valid rows: {unmapped_valid}")
    else:
        print(f"   ✓ All {len(valid_indices):,} valid rows mapped successfully")

    # Verify mapping
    print("\n6. Verifying mapping...")

    assert len(train_indices) == len(dedup_train), "Train mapping size mismatch!"
    assert len(valid_indices) == len(dedup_valid), "Valid mapping size mismatch!"

    # Filter out -1 placeholders for stats
    train_indices_valid = train_indices[train_indices >= 0]
    valid_indices_valid = valid_indices[valid_indices >= 0]

    assert np.max(train_indices_valid) < len(orig_combined), "Train index out of bounds!"
    assert np.max(valid_indices_valid) < len(orig_combined), "Valid index out of bounds!"

    print(f"   ✓ Train indices: min={np.min(train_indices_valid)}, max={np.max(train_indices_valid)}, count={len(train_indices_valid):,}")
    print(f"   ✓ Valid indices: min={np.min(valid_indices_valid)}, max={np.max(valid_indices_valid)}, count={len(valid_indices_valid):,}")
    print(f"   ℹ Combined feature space size: {len(orig_combined):,} rows")

    # Save mapping
    print(f"\n7. Saving mapping to {output_file}...")

    mapping = {
        'train_indices': train_indices,
        'valid_indices': valid_indices,
        'dedup_train_rows': len(dedup_train),
        'dedup_valid_rows': len(dedup_valid),
        'orig_train_rows': len(orig_train),
        'orig_valid_rows': len(orig_valid),
        'combined_feature_space_rows': len(orig_combined),
        'unmapped_train': unmapped_train,
        'unmapped_valid': unmapped_valid
    }

    with open(output_file, 'wb') as f:
        pickle.dump(mapping, f)

    print(f"   Saved mapping with {len(train_indices):,} train and {len(valid_indices):,} valid indices")

    print(f"\n{'='*80}")
    print("MAPPING CREATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nMapping Summary:")
    print(f"  Train: {len(train_indices):,} dedup rows → original row indices")
    print(f"  Valid: {len(valid_indices):,} dedup rows → original row indices")
    print(f"\nUsage in scripts:")
    print(f"  mapping = pickle.load(open('data/index_mapping.pkl', 'rb'))")
    print(f"  train_idx = mapping['train_indices']  # shape: ({len(train_indices):,},)")
    print(f"  valid_idx = mapping['valid_indices']  # shape: ({len(valid_indices):,},)")
    print(f"  X_train = features[train_idx]  # Load only needed features")
    print(f"  X_valid = features[valid_idx]   # Load only needed features\n")

    return mapping


if STRATEGY in ("variant", "both"):
    print("\nCreating variant-based index mapping...")
    create_index_mapping(
        strategy="variant",
        original_train_file=ORIGINAL_TRAIN_FILE,
        original_valid_file=ORIGINAL_VALID_FILE,
        output_dir=OUTPUT_DIR,
    )

if STRATEGY in ("gene", "both"):
    print("\n" + "=" * 80)
    print("Creating gene-based index mapping...")
    print("=" * 80 + "\n")
    create_index_mapping(
        strategy="gene",
        original_train_file=ORIGINAL_TRAIN_FILE,
        original_valid_file=ORIGINAL_VALID_FILE,
        output_dir=OUTPUT_DIR,
    )
