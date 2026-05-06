"""Build the SNV-only train/valid CSVs used for TESSERA SNV pretraining.

Reads the merged TCGA variant table (``TCGA.csv``, produced by
``data/TCGA_PanCan/create_training_data.py``) and the patient-level split
reference (``train_test_patients_reference.csv``, produced by
``create_train_test_patients.py``). Filters to single-nucleotide variants
(``VARIANT_CLASS == 'SNV'``), inner-joins each row to the reference to
inherit the patient-level split, subsamples up to 1,000 variants per
sample (preserving variants seen in at least 5 samples), and writes
``train_data_snv.csv`` / ``valid_data_snv.csv``.

Inputs
------
``TCGA.csv``
    Merged variant table with sample/patient barcodes and clinical type.
``train_test_patients_reference.csv``
    Per-patient split assignment.

Outputs
-------
``train_data_snv.csv``, ``valid_data_snv.csv``
    Per-variant rows with all original columns; ``mut_id`` and the
    intermediate join columns are stripped before writing.

Usage
-----
    python create_training_snv.py \\
        --variants ../../../data/TCGA_PanCan/TCGA.csv \\
        --reference train_test_patients_reference.csv \\
        --output-dir .
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import subsample_variants_with_common

# Columns concatenated to form the per-locus mutation identifier used by the
# subsampler to detect recurrent (cohort-shared) variants.
MUT_ID_COLS = [
    "Hugo_Symbol", "Chromosome", "Start_Position",
    "Reference_Allele", "Tumor_Seq_Allele2",
]

# TCGA patient barcode = first 12 characters of the sample barcode.
PATIENT_BARCODE_LEN = 12

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SNV-only train/valid CSVs for TESSERA pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variants", type=Path,
                        default=Path("../../../data/TCGA_PanCan/TCGA.csv"),
                        help="Path to the merged TCGA variant CSV.")
    parser.add_argument("--reference", type=Path,
                        default=Path("train_test_patients_reference.csv"),
                        help="Path to the patient-level split reference CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for train_data_snv.csv / valid_data_snv.csv.")
    parser.add_argument("--subsample-size", type=int, default=1000,
                        help="Maximum variants kept per sample.")
    parser.add_argument("--min-shared", type=int, default=5,
                        help="Minimum samples a variant must appear in to be "
                             "preserved as recurrent.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the within-sample subsample.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.variants.exists():
        raise FileNotFoundError(f"Variant CSV not found: {args.variants}")
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference CSV not found: {args.reference}")

    logger.info("Loading reference: %s", args.reference)
    df_ref = pd.read_csv(args.reference)
    logger.info("Reference patients: %d", len(df_ref))

    logger.info("Loading variants: %s", args.variants)
    data = pd.read_csv(args.variants)
    logger.info("Raw variants: %d", len(data))

    data = data[data["VARIANT_CLASS"] == "SNV"].copy()
    logger.info("After SNV filter: %d variants", len(data))

    data["patient_barcode"] = data["Tumor_Sample_Barcode"].str[:PATIENT_BARCODE_LEN]
    data = data.merge(
        df_ref[["bcr_patient_barcode", "split"]],
        left_on="patient_barcode", right_on="bcr_patient_barcode", how="inner",
    )
    logger.info("After reference inner-join: %d variants from %d samples",
                len(data), data["Tumor_Sample_Barcode"].nunique())

    data["mut_id"] = data[MUT_ID_COLS].astype(str).agg("_".join, axis=1)
    data = subsample_variants_with_common(
        data,
        subsample_size=args.subsample_size,
        min_shared=args.min_shared,
        random_state=args.seed,
    )
    logger.info("After subsample (n<=%d, min_shared=%d): %d variants from %d samples",
                args.subsample_size, args.min_shared,
                len(data), data["Tumor_Sample_Barcode"].nunique())

    train_data = data[data["split"] == "train"].copy()
    valid_data = data[data["split"] == "test"].copy()
    drop_cols = ["patient_barcode", "bcr_patient_barcode", "split", "mut_id"]
    for df in (train_data, valid_data):
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_data_snv.csv"
    valid_path = args.output_dir / "valid_data_snv.csv"
    train_data.to_csv(train_path, index=False)
    valid_data.to_csv(valid_path, index=False)
    logger.info("Wrote %d train rows -> %s", len(train_data), train_path)
    logger.info("Wrote %d valid rows -> %s", len(valid_data), valid_path)


if __name__ == "__main__":
    main(parse_args())
