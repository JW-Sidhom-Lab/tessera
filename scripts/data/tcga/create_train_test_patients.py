"""Build the patient-level train/test reference split for TCGA pretraining.

Reads the TCGA Pan-Cancer Clinical Data Resource (``clinical.csv``), filters
to the 31 solid-tumor types used in the manuscript, performs a
tumor-type-stratified 75/25 patient-level split (``random_state=42``), and
writes the result as ``train_test_patients_reference.csv``. Downstream
``create_training_*.py`` scripts join their per-sample tables to this file to
inherit the patient-level split assignment, ensuring no patient appears in
both training and validation across modalities.

Inputs
------
``clinical.csv``
    Patient-level table from the TCGA-CDR ``ExtraEndpoints`` sheet
    (columns ``bcr_patient_barcode``, ``type``, plus outcomes columns).

Output
------
``train_test_patients_reference.csv``
    Columns ``bcr_patient_barcode``, ``type``, ``split`` (``"train"`` or
    ``"test"``).

Usage
-----
    python create_train_test_patients.py \\
        --clinical ../../../data/TCGA_PanCan/clinical.csv \\
        --output train_test_patients_reference.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 31 solid tumor types used for TESSERA pretraining (excludes hematologic
# cancers like LAML, DLBC).
SOLID_TUMOR_TYPES = [
    "UCEC", "SKCM", "COAD", "LUAD", "STAD", "LUSC", "BLCA", "BRCA", "HNSC",
    "CESC", "GBM", "READ", "LIHC", "OV", "ESCA", "LGG", "PAAD", "PRAD",
    "SARC", "KIRP", "KIRC", "THCA", "ACC", "UCS", "THYM", "MESO", "CHOL",
    "KICH", "TGCT", "PCPG", "UVM",
]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the patient-level train/test reference for TCGA pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clinical", type=Path,
                        default=Path("../../../data/TCGA_PanCan/clinical.csv"),
                        help="Path to TCGA-CDR clinical CSV.")
    parser.add_argument("--output", type=Path,
                        default=Path("train_test_patients_reference.csv"),
                        help="Output CSV path.")
    parser.add_argument("--test-frac", type=float, default=0.25,
                        help="Fraction of patients held out for validation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the stratified split.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.clinical.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {args.clinical}")

    logger.info("Loading clinical: %s", args.clinical)
    df_clin = pd.read_csv(args.clinical)
    df_clin.columns = [c.lstrip("﻿") for c in df_clin.columns]
    if not {"bcr_patient_barcode", "type"}.issubset(df_clin.columns):
        raise ValueError(
            f"Clinical CSV must contain 'bcr_patient_barcode' and 'type'; "
            f"got {list(df_clin.columns)}"
        )

    df_patients = df_clin[["bcr_patient_barcode", "type"]].drop_duplicates()
    logger.info("Unique patients: %d", len(df_patients))

    n_before = len(df_patients)
    df_patients = df_patients[df_patients["type"].isin(SOLID_TUMOR_TYPES)].copy()
    logger.info("Filtered to %d solid-tumor patients (dropped %d non-solid)",
                len(df_patients), n_before - len(df_patients))

    train, test = train_test_split(
        df_patients,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=df_patients["type"],
    )
    train["split"] = "train"
    test["split"] = "test"
    df_ref = pd.concat([train, test], ignore_index=True)
    df_ref = df_ref.sort_values("bcr_patient_barcode").reset_index(drop=True)
    logger.info("Split: train=%d, test=%d", len(train), len(test))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_ref.to_csv(args.output, index=False)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main(parse_args())
