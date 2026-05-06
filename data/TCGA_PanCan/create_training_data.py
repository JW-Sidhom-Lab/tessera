"""Convert TCGA MC3 somatic mutations into the variant CSV used for TESSERA pretraining.

Reads the TCGA Pan-Cancer MC3 public MAF, computes per-variant variant-allele
frequency (VAF), joins each row to the patient's tumor-type acronym,
deduplicates variants reported by multiple MC3 callers at the same locus, caps
overly long indel allele strings, and writes the result as a comma-separated
table. Downstream pretraining scripts split this table into train/valid
partitions and filter to SNV-only when needed.

Inputs
------
``mc3.v0.2.8.PUBLIC.maf``
    TCGA Pan-Cancer somatic mutation call set (MC3; Ellrott et al., Cell
    Systems 2018). Tab-separated, one row per call per caller.
``clinical.csv``
    Patient -> tumor-type acronym mapping, exported from the
    ``ExtraEndpoints`` sheet of the TCGA Pan-Cancer Clinical Data Resource
    (Liu et al., Cell 2018; ``TCGA-CDR-SupplementalTableS1.xlsx``).

Both files are distributed together on the TCGA Pan-Cancer Atlas downloads
page: https://gdc.cancer.gov/about-data/publications/pancanatlas

Output
------
``TCGA.csv``
    Variant table with columns ``Tumor_Sample_Barcode, Hugo_Symbol, Chromosome,
    Start_Position, Reference_Allele, Tumor_Seq_Allele2, vaf, VARIANT_CLASS,
    HGVSp_Short, bcr_patient_barcode, type``. Includes both SNVs and indels.

Usage
-----
    python create_training_data.py \\
        --maf mc3.v0.2.8.PUBLIC.maf \\
        --clinical clinical.csv \\
        --output TCGA.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# Cap on Reference_Allele / Tumor_Seq_Allele2 string lengths. Long indels are
# truncated to ALLELE_FRONT_LEN leading + ALLELE_BACK_LEN trailing bases so they
# fit TESSERA's fixed-length per-token allele encoder; do not change without
# updating the model's allele vocabulary.
ALLELE_FRONT_LEN = 5
ALLELE_BACK_LEN = 5

# TCGA barcode hierarchy: ``TCGA-XX-NNNN`` (12 chars) is the patient ID,
# ``TCGA-XX-NNNN-NN`` (15 chars) is the sample (vial-level) ID and is the
# natural unit for somatic-variant aggregation.
PATIENT_BARCODE_LEN = 12
SAMPLE_BARCODE_LEN = 15

KEEP_COLS = [
    "Tumor_Sample_Barcode",
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "vaf",
    "VARIANT_CLASS",
    "HGVSp_Short",
]

# Locus-uniqueness keys for deduplication. MC3 emits one row per caller, so the
# same variant can appear up to seven times in the raw MAF.
DEDUP_KEYS = [
    "Tumor_Sample_Barcode",
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
]

logger = logging.getLogger(__name__)


def truncate_allele(allele: str,
                    front: int = ALLELE_FRONT_LEN,
                    back: int = ALLELE_BACK_LEN) -> str:
    """Cap an allele string to ``front`` leading + ``back`` trailing bases."""
    if len(allele) > front + back:
        return allele[:front] + allele[-back:]
    return allele


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TCGA MC3 MAF into the TESSERA variant CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--maf", type=Path, default=Path("mc3.v0.2.8.PUBLIC.maf"),
                        help="Path to TCGA MC3 public MAF (tab-separated).")
    parser.add_argument("--clinical", type=Path, default=Path("clinical.csv"),
                        help="Path to TCGA-CDR clinical CSV "
                             "(must contain bcr_patient_barcode and type).")
    parser.add_argument("--output", type=Path, default=Path("TCGA.csv"),
                        help="Output path for the variant CSV.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.maf.exists():
        raise FileNotFoundError(f"MAF not found: {args.maf}")
    if not args.clinical.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {args.clinical}")

    logger.info("Loading MAF: %s", args.maf)
    df_var = pd.read_csv(args.maf, sep="\t", low_memory=False)
    logger.info("MAF rows: %d", len(df_var))

    required = {"t_depth", "t_alt_count", "t_ref_count"} | (set(KEEP_COLS) - {"vaf"})
    missing = required - set(df_var.columns)
    if missing:
        raise ValueError(f"MAF is missing required columns: {sorted(missing)}")

    n_before = len(df_var)
    df_var = df_var[df_var["t_depth"] != 0].copy()
    logger.info("Dropped %d rows with t_depth==0 (kept %d)",
                n_before - len(df_var), len(df_var))

    alt = df_var["t_alt_count"].astype(int)
    ref = df_var["t_ref_count"].astype(int)
    df_var["vaf"] = alt / (alt + ref)

    df_var = df_var[KEEP_COLS].copy()
    df_var["bcr_patient_barcode"] = df_var["Tumor_Sample_Barcode"].str[:PATIENT_BARCODE_LEN]
    df_var["Tumor_Sample_Barcode"] = df_var["Tumor_Sample_Barcode"].str[:SAMPLE_BARCODE_LEN]

    logger.info("Loading clinical: %s", args.clinical)
    df_clin = pd.read_csv(args.clinical)
    df_clin.columns = [c.lstrip("﻿") for c in df_clin.columns]
    if not {"bcr_patient_barcode", "type"}.issubset(df_clin.columns):
        raise ValueError(
            f"Clinical CSV must contain 'bcr_patient_barcode' and 'type' columns; "
            f"got {list(df_clin.columns)}"
        )

    df_out = df_var.merge(
        df_clin[["bcr_patient_barcode", "type"]],
        how="left",
        on="bcr_patient_barcode",
    )
    n_unmapped = df_out["type"].isna().sum()
    if n_unmapped:
        logger.warning("%d variant rows had no tumor type after clinical merge",
                       n_unmapped)

    n_before = len(df_out)
    df_out = df_out.drop_duplicates(subset=DEDUP_KEYS).reset_index(drop=True)
    logger.info("Deduplicated %d -> %d rows", n_before, len(df_out))

    df_out["Reference_Allele"] = df_out["Reference_Allele"].astype(str).map(truncate_allele)
    df_out["Tumor_Seq_Allele2"] = df_out["Tumor_Seq_Allele2"].astype(str).map(truncate_allele)

    logger.info("Writing %d rows to %s", len(df_out), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main(parse_args())
