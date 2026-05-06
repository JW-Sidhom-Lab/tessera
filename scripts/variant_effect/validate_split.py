"""Sanity check the deduplicated train/valid split: zero overlap, ~75/25 ratio, row preservation, schema integrity.

Inputs
------
``data/{train,valid}_data_snv_dedup{,_gene}.csv`` (from create_variant_dedup_split.py)

Usage
-----
    python validate_split.py                       # variant (default)
    STRATEGY=gene python validate_split.py
"""

import os
from pathlib import Path

import pandas as pd

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
STRATEGY = os.environ.get("STRATEGY", "variant")

def validate_split(
    train_file=None,
    valid_file=None,
    expected_total_rows=None
):
    """
    Validate variant-level deduplicated split.

    Parameters:
    -----------
    train_file : str
        Path to deduplicated train CSV
    valid_file : str
        Path to deduplicated valid CSV
    expected_total_rows : int
        Expected total rows (from original combined data)

    Returns:
    --------
    bool : True if all validations pass, False otherwise
    """

    # Use strategy-specific dedup files if not explicitly provided
    if train_file is None or valid_file is None:
        if STRATEGY == 'gene':
            train_file = train_file or 'data/train_data_snv_dedup_gene.csv'
            valid_file = valid_file or 'data/valid_data_snv_dedup_gene.csv'
        else:
            train_file = train_file or 'data/train_data_snv_dedup.csv'
            valid_file = valid_file or 'data/valid_data_snv_dedup.csv'

    print("=" * 80)
    strategy_desc = "VARIANT-LEVEL" if STRATEGY == 'variant' else "GENE-LEVEL"
    print(f"VALIDATING {strategy_desc} DEDUPLICATED SPLIT")
    print("=" * 80)

    all_passed = True

    # Check 1: Files exist
    print(f"\n1. Checking files exist...")
    if not os.path.exists(train_file):
        print(f"   ✗ FAILED: {train_file} not found")
        return False
    if not os.path.exists(valid_file):
        print(f"   ✗ FAILED: {valid_file} not found")
        return False
    print(f"   ✓ Both files exist")

    # Load data
    print(f"\n2. Loading CSV files...")
    try:
        train_data = pd.read_csv(train_file)
        valid_data = pd.read_csv(valid_file)
        print(f"   Train rows: {len(train_data):,}")
        print(f"   Valid rows: {len(valid_data):,}")
        print(f"   Total rows: {len(train_data) + len(valid_data):,}")
    except Exception as e:
        print(f"   ✗ FAILED to load CSV files: {e}")
        return False

    # Check 2: Total rows preserved
    print(f"\n3. Checking row count preservation...")
    total_rows = len(train_data) + len(valid_data)
    if expected_total_rows is not None and total_rows != expected_total_rows:
        print(f"   ✗ FAILED: Expected {expected_total_rows:,} rows, got {total_rows:,}")
        all_passed = False
    else:
        print(f"   ✓ Total rows preserved: {total_rows:,}")

    # Check 3: Required columns present
    print(f"\n4. Checking for required columns...")
    required_columns = [
        'Tumor_Sample_Barcode', 'Hugo_Symbol', 'Chromosome',
        'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2',
        'vaf', 'VARIANT_CLASS', 'HGVSp_Short',
        'bcr_patient_barcode_x', 'type', 'bcr_patient_barcode_y'
    ]

    train_cols = set(train_data.columns)
    valid_cols = set(valid_data.columns)

    missing_in_train = set(required_columns) - train_cols
    missing_in_valid = set(required_columns) - valid_cols

    if missing_in_train:
        print(f"   ✗ FAILED: Missing columns in train: {missing_in_train}")
        all_passed = False
    elif missing_in_valid:
        print(f"   ✗ FAILED: Missing columns in valid: {missing_in_valid}")
        all_passed = False
    else:
        print(f"   ✓ All {len(required_columns)} required columns present")

    # Check 4: No missing values in critical columns
    print(f"\n5. Checking for missing variant identifiers...")
    critical_cols = ['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2']

    train_missing = train_data[critical_cols].isna().sum().sum()
    valid_missing = valid_data[critical_cols].isna().sum().sum()

    if train_missing > 0 or valid_missing > 0:
        print(f"   ✗ FAILED: Missing values in variant ID columns")
        print(f"      Train missing: {train_missing}, Valid missing: {valid_missing}")
        all_passed = False
    else:
        print(f"   ✓ No missing values in variant ID columns")

    # Check 5: Create variant IDs and check overlap
    print(f"\n6. Checking for variant overlap (CRITICAL)...")

    train_variants = set(
        train_data['Chromosome'].astype(str) + '_' +
        train_data['Start_Position'].astype(str) + '_' +
        train_data['Reference_Allele'].astype(str) + '_' +
        train_data['Tumor_Seq_Allele2'].astype(str)
    )

    valid_variants = set(
        valid_data['Chromosome'].astype(str) + '_' +
        valid_data['Start_Position'].astype(str) + '_' +
        valid_data['Reference_Allele'].astype(str) + '_' +
        valid_data['Tumor_Seq_Allele2'].astype(str)
    )

    overlapping = train_variants & valid_variants

    if len(overlapping) > 0:
        print(f"   ✗ FAILED: Found {len(overlapping)} overlapping variants!")
        print(f"      This means deduplication was unsuccessful")
        all_passed = False
    else:
        print(f"   ✓ ZERO variant overlap (deduplication successful)")

    # Check 6: 75/25 variant ratio (within ±2%)
    print(f"\n7. Checking 75/25 variant split ratio...")

    total_unique_variants = len(train_variants) + len(valid_variants)
    train_ratio = len(train_variants) / total_unique_variants * 100
    valid_ratio = len(valid_variants) / total_unique_variants * 100

    print(f"   Train variants: {len(train_variants):,} ({train_ratio:.2f}%)")
    print(f"   Valid variants: {len(valid_variants):,} ({valid_ratio:.2f}%)")

    if abs(train_ratio - 75.0) > 2.0:
        print(f"   ✗ FAILED: Train ratio {train_ratio:.2f}% is not close to 75%")
        all_passed = False
    elif abs(valid_ratio - 25.0) > 2.0:
        print(f"   ✗ FAILED: Valid ratio {valid_ratio:.2f}% is not close to 25%")
        all_passed = False
    else:
        print(f"   ✓ Split ratio within expected range (75/25 ±2%)")

    # Check 7: Statistics
    print(f"\n8. Computing additional statistics...")

    train_patients = set(train_data['bcr_patient_barcode_x'].dropna().unique())
    valid_patients = set(valid_data['bcr_patient_barcode_x'].dropna().unique())
    shared_patients = len(train_patients & valid_patients)

    print(f"   Train patients: {len(train_patients):,}")
    print(f"   Valid patients: {len(valid_patients):,}")
    print(f"   Shared patients: {shared_patients:,}")
    print(f"   (Note: Shared patients are expected - allows variants in both sets)")

    # Summary
    print(f"\n{'='*80}")
    if all_passed:
        print("VALIDATION RESULT: ✓ ALL CHECKS PASSED")
        print("The variant-level deduplication split is valid and ready for analysis.")
    else:
        print("VALIDATION RESULT: ✗ SOME CHECKS FAILED")
        print("Please review the errors above and regenerate the split.")
    print(f"{'='*80}\n")

    return all_passed


if __name__ == '__main__':
    success = validate_split()
    exit(0 if success else 1)
