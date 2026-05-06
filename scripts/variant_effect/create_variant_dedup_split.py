"""Build the variant-level or gene-level deduplicated train/valid split for ClinVar variant-effect prediction.

Reads the patient-level TCGA SNV split from
``../data/tcga/{train,valid}_data_snv.csv`` (produced by
``scripts/data/tcga/create_training_snv.py``) and re-splits 75/25 such
that each unique variant (variant strategy) or each unique gene (gene
strategy) appears exclusively in either the train or validation set.
This eliminates the "model memorised the variant" failure mode for the
variant-effect prediction evaluation in Figure 1 h-o.

The variant strategy backs Figure 1 h-k; the gene strategy backs Figure
1 l-o.

Inputs
------
``../data/tcga/train_data_snv.csv``, ``../data/tcga/valid_data_snv.csv``

Outputs
-------
``data/train_data_snv_dedup{,_gene}.csv``
``data/valid_data_snv_dedup{,_gene}.csv``
``data/split_report{,_gene}.txt``

Usage
-----
    python create_variant_dedup_split.py                       # both strategies (default)
    STRATEGY=variant python create_variant_dedup_split.py      # variant only
    STRATEGY=gene    python create_variant_dedup_split.py      # gene only
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
# 'variant', 'gene', or 'both' (default; runs both strategies).
STRATEGY = os.environ.get("STRATEGY", "both")
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA", "../data/tcga/train_data_snv.csv")
VALID_DATA_PATH = os.environ.get("VALID_DATA", "../data/tcga/valid_data_snv.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.75"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

def create_variant_dedup_split(
    train_file='../data/tcga/train_data_snv.csv',
    valid_file='../data/tcga/valid_data_snv.csv',
    output_dir='data',
    train_ratio=0.75,
    random_seed=42,
    strategy='variant'
):
    """
    Create deduplicated train/validation split with configurable strategy.

    Parameters:
    -----------
    train_file : str
        Path to original patient-level train CSV
    valid_file : str
        Path to original patient-level valid CSV
    output_dir : str
        Directory to save deduplicated CSVs
    train_ratio : float
        Ratio of elements to assign to training set (default: 0.75)
    random_seed : int
        Random seed for reproducibility (default: 42)
    strategy : str
        Deduplication strategy: 'variant' or 'gene' (default: 'variant')

    Returns:
    --------
    dict : Statistics about the split
    """

    print("=" * 80)
    strategy_desc = "VARIANT-LEVEL" if strategy == 'variant' else "GENE-LEVEL"
    print(f"CREATING {strategy_desc} DEDUPLICATED TRAIN/VALIDATION SPLIT")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print(f"\n1. Loading data from:")
    print(f"   Train: {train_file}")
    print(f"   Valid: {valid_file}")

    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)

    print(f"   Train rows: {len(train_data):,}")
    print(f"   Valid rows: {len(valid_data):,}")
    print(f"   Total rows: {len(train_data) + len(valid_data):,}")

    # Step 2: Combine datasets and create IDs (variant or gene based on strategy)
    if strategy == 'gene':
        print(f"\n2. Creating gene IDs (Hugo_Symbol)...")
    else:
        print(f"\n2. Creating variant IDs (chr_pos_ref_alt)...")

    # Add source column to track original split
    train_data['original_split'] = 'train'
    valid_data['original_split'] = 'valid'

    # Combine
    combined_data = pd.concat([train_data, valid_data], ignore_index=True)

    if strategy == 'gene':
        # Handle missing gene names - assign to 'UNKNOWN' pseudo-gene
        combined_data['Hugo_Symbol'] = combined_data['Hugo_Symbol'].fillna('UNKNOWN')
        combined_data['split_id'] = combined_data['Hugo_Symbol']
        n_unique = combined_data['split_id'].nunique()
        print(f"   Total rows after combining: {len(combined_data):,}")
        print(f"   Unique genes: {n_unique:,}")
        if (combined_data['split_id'] == 'UNKNOWN').sum() > 0:
            print(f"   (includes {(combined_data['split_id'] == 'UNKNOWN').sum():,} rows with unknown gene)")
    else:  # variant strategy
        # Create variant ID
        combined_data['var_id'] = (
            combined_data['Chromosome'].astype(str) + '_' +
            combined_data['Start_Position'].astype(int).astype(str) + '_' +
            combined_data['Reference_Allele'].astype(str) + '_' +
            combined_data['Tumor_Seq_Allele2'].astype(str)
        )
        combined_data['split_id'] = combined_data['var_id']
        n_unique = combined_data['split_id'].nunique()
        print(f"   Total rows after combining: {len(combined_data):,}")
        print(f"   Unique variants: {n_unique:,}")

    # Step 3: Analyze current overlap in original patient-level split
    if strategy == 'gene':
        print(f"\n3. Analyzing gene overlap in original patient-level split...")
        genes_in_train = set(train_data['Hugo_Symbol'].fillna('UNKNOWN').unique())
        genes_in_valid = set(valid_data['Hugo_Symbol'].fillna('UNKNOWN').unique())
        overlapping = genes_in_train & genes_in_valid
        print(f"   Genes ONLY in train: {len(genes_in_train - genes_in_valid):,}")
        print(f"   Genes ONLY in valid: {len(genes_in_valid - genes_in_train):,}")
        print(f"   Genes in BOTH sets: {len(overlapping):,}")
        if len(genes_in_train) > 0:
            print(f"   Overlap percentage: {len(overlapping) / len(genes_in_train) * 100:.2f}%")
    else:
        print(f"\n3. Analyzing variant overlap in original patient-level split...")
        variants_in_train = set(train_data['Chromosome'].astype(str) + '_' +
                               train_data['Start_Position'].astype(int).astype(str) + '_' +
                               train_data['Reference_Allele'].astype(str) + '_' +
                               train_data['Tumor_Seq_Allele2'].astype(str))

        variants_in_valid = set(valid_data['Chromosome'].astype(str) + '_' +
                               valid_data['Start_Position'].astype(int).astype(str) + '_' +
                               valid_data['Reference_Allele'].astype(str) + '_' +
                               valid_data['Tumor_Seq_Allele2'].astype(str))

        overlapping_variants = variants_in_train & variants_in_valid
        print(f"   Variants ONLY in train: {len(variants_in_train - variants_in_valid):,}")
        print(f"   Variants ONLY in valid: {len(variants_in_valid - variants_in_train):,}")
        print(f"   Variants in BOTH sets: {len(overlapping_variants):,}")
        print(f"   Overlap percentage: {len(overlapping_variants) / len(variants_in_train) * 100:.2f}%")

    # Step 4: Get unique elements and create 75/25 split
    if strategy == 'gene':
        print(f"\n4. Creating gene-level 75/25 split (seed={random_seed})...")
        element_name = 'genes'
    else:
        print(f"\n4. Creating variant-level 75/25 split (seed={random_seed})...")
        element_name = 'variants'

    unique_elements = combined_data['split_id'].unique()
    n_total_elements = len(unique_elements)
    n_train_elements = int(n_total_elements * train_ratio)

    print(f"   Total unique {element_name}: {n_total_elements:,}")
    print(f"   {element_name.capitalize()} for training ({train_ratio*100:.0f}%): {n_train_elements:,}")
    print(f"   {element_name.capitalize()} for validation ({(1-train_ratio)*100:.0f}%): {n_total_elements - n_train_elements:,}")

    # Randomly shuffle and split
    np.random.shuffle(unique_elements)
    train_element_set = set(unique_elements[:n_train_elements])
    valid_element_set = set(unique_elements[n_train_elements:])

    # Verify no overlap
    overlap_check = train_element_set & valid_element_set
    assert len(overlap_check) == 0, f"ERROR: Found {len(overlap_check)} overlapping {element_name}!"

    print(f"   ✓ {element_name.capitalize()} overlap check PASSED (zero overlap)")

    # Step 5: Assign all rows to their element's designated split
    if strategy == 'gene':
        print(f"\n5. Assigning all rows to gene-assigned splits...")
    else:
        print(f"\n5. Assigning all rows to variant-assigned splits...")

    combined_data['dedup_split'] = combined_data['split_id'].apply(
        lambda element_id: 'train' if element_id in train_element_set else 'valid'
    )

    train_data_dedup = combined_data[combined_data['dedup_split'] == 'train'].copy()
    valid_data_dedup = combined_data[combined_data['dedup_split'] == 'valid'].copy()

    print(f"   Train rows in new split: {len(train_data_dedup):,}")
    print(f"   Valid rows in new split: {len(valid_data_dedup):,}")
    print(f"   Total rows preserved: {len(train_data_dedup) + len(valid_data_dedup):,}")

    # Verify total rows preserved
    assert len(train_data_dedup) + len(valid_data_dedup) == len(combined_data), \
        "ERROR: Row count mismatch!"
    print(f"   ✓ Row count check PASSED (all {len(combined_data):,} rows preserved)")

    # Step 6: Remove temporary columns and save
    print(f"\n6. Saving deduplicated CSVs...")

    # Remove temporary columns
    cols_to_remove = ['original_split', 'dedup_split', 'split_id']
    if 'var_id' in train_data_dedup.columns:
        cols_to_remove.append('var_id')

    columns_to_keep = [col for col in train_data_dedup.columns if col not in cols_to_remove]

    train_data_dedup = train_data_dedup[columns_to_keep]
    valid_data_dedup = valid_data_dedup[columns_to_keep]

    # Save files with strategy-specific names
    if strategy == 'gene':
        train_output = os.path.join(output_dir, 'train_data_snv_dedup_gene.csv')
        valid_output = os.path.join(output_dir, 'valid_data_snv_dedup_gene.csv')
    else:
        train_output = os.path.join(output_dir, 'train_data_snv_dedup.csv')
        valid_output = os.path.join(output_dir, 'valid_data_snv_dedup.csv')

    train_data_dedup.to_csv(train_output, index=False)
    valid_data_dedup.to_csv(valid_output, index=False)

    print(f"   Saved: {train_output}")
    print(f"   Saved: {valid_output}")

    # Step 7: Generate split report
    print(f"\n7. Generating split report...")

    # Get unique variants/genes in new split
    if strategy == 'gene':
        train_unique_elements = train_data_dedup['Hugo_Symbol'].fillna('UNKNOWN').nunique()
        valid_unique_elements = valid_data_dedup['Hugo_Symbol'].fillna('UNKNOWN').nunique()
        total_unique_elements = train_unique_elements + valid_unique_elements
        element_label = "GENES"
        element_singular = "gene"
        report_title = "GENE-LEVEL DEDUPLICATION SPLIT REPORT"
        report_file_suffix = 'split_report_gene.txt'
    else:
        train_unique_elements = train_data_dedup['Chromosome'].astype(str).str.cat(
            [train_data_dedup['Start_Position'].astype(str),
             train_data_dedup['Reference_Allele'].astype(str),
             train_data_dedup['Tumor_Seq_Allele2'].astype(str)],
            sep='_'
        ).nunique()

        valid_unique_elements = valid_data_dedup['Chromosome'].astype(str).str.cat(
            [valid_data_dedup['Start_Position'].astype(str),
             valid_data_dedup['Reference_Allele'].astype(str),
             valid_data_dedup['Tumor_Seq_Allele2'].astype(str)],
            sep='_'
        ).nunique()
        total_unique_elements = train_unique_elements + valid_unique_elements
        element_label = "VARIANTS"
        element_singular = "variant"
        report_title = "VARIANT-LEVEL DEDUPLICATION SPLIT REPORT"
        report_file_suffix = 'split_report.txt'

    # Calculate patient overlap
    train_patients = set(train_data_dedup['bcr_patient_barcode_x'].dropna().unique())
    valid_patients = set(valid_data_dedup['bcr_patient_barcode_x'].dropna().unique())
    shared_patients = len(train_patients & valid_patients)

    report = f"""
{report_title}
{'='*60}

SPLIT SIZES:
  Train rows:        {len(train_data_dedup):>12,}
  Valid rows:        {len(valid_data_dedup):>12,}
  Total rows:        {len(train_data_dedup) + len(valid_data_dedup):>12,}

UNIQUE {element_label}:
  Train {element_singular}s:    {train_unique_elements:>12,}
  Valid {element_singular}s:    {valid_unique_elements:>12,}
  Total {element_singular}s:    {total_unique_elements:>12,}
  Overlapping:       {0:>12,} (0.00%)  ✓ ZERO LEAKAGE

{element_label} RATIO:
  Train %:           {train_unique_elements / total_unique_elements * 100:>11.2f}%
  Valid %:           {valid_unique_elements / total_unique_elements * 100:>11.2f}%

PATIENT STATISTICS:
  Train patients:    {len(train_patients):>12,}
  Valid patients:    {len(valid_patients):>12,}
  Shared patients:   {shared_patients:>12,} (same patient can have {element_singular}s in both sets)

DATA INTEGRITY CHECKS:
✓ Total rows preserved: {len(train_data_dedup) + len(valid_data_dedup):,}
✓ Zero {element_singular} overlap (deduplication successful)
✓ Random seed: {random_seed}

FILES CREATED:
- {train_output}
- {valid_output}
- {os.path.join(output_dir, report_file_suffix)}
"""

    report_file = os.path.join(output_dir, report_file_suffix)
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"   Saved: {report_file}")
    print(report)

    print(f"\n{'='*80}")
    print("DATA SPLIT COMPLETE - READY FOR ANALYSIS")
    print(f"{'='*80}\n")

    # Return stats with appropriate key names based on strategy
    if strategy == 'gene':
        return {
            'train_rows': len(train_data_dedup),
            'valid_rows': len(valid_data_dedup),
            'train_genes': train_unique_elements,
            'valid_genes': valid_unique_elements,
            'overlapping_genes': 0,
            'train_patients': len(train_patients),
            'valid_patients': len(valid_patients),
            'shared_patients': shared_patients,
            'strategy': 'gene'
        }
    else:
        return {
            'train_rows': len(train_data_dedup),
            'valid_rows': len(valid_data_dedup),
            'train_variants': train_unique_elements,
            'valid_variants': valid_unique_elements,
            'overlapping_variants': 0,
            'train_patients': len(train_patients),
            'valid_patients': len(valid_patients),
            'shared_patients': shared_patients,
            'strategy': 'variant'
        }


if STRATEGY in ("variant", "both"):
    print("\nCreating variant-based deduplication...")
    create_variant_dedup_split(
        train_file=TRAIN_DATA_PATH, valid_file=VALID_DATA_PATH,
        output_dir=OUTPUT_DIR, train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED, strategy="variant",
    )

if STRATEGY in ("gene", "both"):
    print("\n" + "=" * 80)
    print("Creating gene-based deduplication...")
    print("=" * 80 + "\n")
    create_variant_dedup_split(
        train_file=TRAIN_DATA_PATH, valid_file=VALID_DATA_PATH,
        output_dir=OUTPUT_DIR, train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED, strategy="gene",
    )
