"""Build the CNA train/valid CSVs used for TESSERA CNA pretraining.

Reads the TCGA CNA segment table (``TCGA_CNA.csv``, produced by
``data/TCGA_PanCan/create_training_data_cna.py``) and the patient-level
split reference, inner-joins each segment to the reference to inherit
the patient-level split, subsamples up to 1,000 segments per sample by
alteration magnitude (``|Segment_Mean| + 0.5 * LOH``), and writes
``train_data_cna.csv`` / ``valid_data_cna.csv``.

Inputs
------
``TCGA_CNA.csv``
    Per-segment CNA table with columns ``Sample, Chromosome, Start, End,
    Segment_Mean, LOH``. ``Sample`` is renamed to ``Tumor_Sample_Barcode``.
``train_test_patients_reference.csv``
    Per-patient split assignment.

Outputs
-------
``train_data_cna.csv``, ``valid_data_cna.csv``
    Per-segment rows with intermediate join columns stripped.

Usage
-----
    python create_training_cna.py \\
        --cna ../../../data/TCGA_PanCan/TCGA_CNA.csv \\
        --reference train_test_patients_reference.csv \\
        --output-dir .
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import subsample_cna_priority

# TCGA patient barcode = first 12 characters of the sample barcode.
PATIENT_BARCODE_LEN = 12

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CNA train/valid CSVs for TESSERA pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cna", type=Path,
                        default=Path("../../../data/TCGA_PanCan/TCGA_CNA.csv"),
                        help="Path to the TCGA CNA segment CSV.")
    parser.add_argument("--reference", type=Path,
                        default=Path("train_test_patients_reference.csv"),
                        help="Path to the patient-level split reference CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for train_data_cna.csv / valid_data_cna.csv.")
    parser.add_argument("--subsample-size", type=int, default=1000,
                        help="Maximum CNA segments kept per sample.")
    parser.add_argument("--loh-weight", type=float, default=0.5,
                        help="Bonus added to |Segment_Mean| for LOH segments.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.cna.exists():
        raise FileNotFoundError(f"CNA CSV not found: {args.cna}")
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference CSV not found: {args.reference}")

    logger.info("Loading reference: %s", args.reference)
    df_ref = pd.read_csv(args.reference)

    logger.info("Loading CNA: %s", args.cna)
    df_cna = pd.read_csv(args.cna)
    logger.info("Raw CNA segments: %d from %d samples",
                len(df_cna), df_cna["Sample"].nunique())

    df_cna = df_cna.rename(columns={"Sample": "Tumor_Sample_Barcode"})
    df_cna["patient_barcode"] = df_cna["Tumor_Sample_Barcode"].str[:PATIENT_BARCODE_LEN]
    df_cna = df_cna.merge(
        df_ref[["bcr_patient_barcode", "split"]],
        left_on="patient_barcode", right_on="bcr_patient_barcode", how="inner",
    )
    logger.info("After reference inner-join: %d segments from %d samples",
                len(df_cna), df_cna["Tumor_Sample_Barcode"].nunique())

    df_cna = subsample_cna_priority(
        df_cna,
        subsample_size=args.subsample_size,
        loh_weight=args.loh_weight,
        sample_col="Tumor_Sample_Barcode",
    )
    logger.info("After subsample (n<=%d, loh_weight=%.2f): %d segments from %d samples",
                args.subsample_size, args.loh_weight,
                len(df_cna), df_cna["Tumor_Sample_Barcode"].nunique())

    train_cna = df_cna[df_cna["split"] == "train"].copy()
    valid_cna = df_cna[df_cna["split"] == "test"].copy()
    drop_cols = ["patient_barcode", "bcr_patient_barcode", "split"]
    for df in (train_cna, valid_cna):
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_data_cna.csv"
    valid_path = args.output_dir / "valid_data_cna.csv"
    train_cna.to_csv(train_path, index=False)
    valid_cna.to_csv(valid_path, index=False)
    logger.info("Wrote %d train rows -> %s", len(train_cna), train_path)
    logger.info("Wrote %d valid rows -> %s", len(valid_cna), valid_path)


if __name__ == "__main__":
    main(parse_args())
