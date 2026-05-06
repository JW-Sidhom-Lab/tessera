"""Convert GENIE somatic mutations into the variant CSV used for TESSERA cross-platform SNV validation.

Reads the AACR Project GENIE v18.0 public release MAF, computes per-variant
variant-allele frequency (VAF), deduplicates calls, joins each row to its
sample-level clinical metadata (``ONCOTREE_CODE``, ``SEQ_ASSAY_ID``, etc.),
caps overly long indel allele strings, and writes the result as a comma-
separated table. Downstream SNV validation scripts split this table into
train/valid partitions and filter to SNV-only when needed.

Inputs
------
``data_mutations_extended.txt``
    GENIE mutations MAF (cBioPortal/GENIE format). Tab-separated. Required
    columns: ``Tumor_Sample_Barcode, Hugo_Symbol, Chromosome,
    Start_Position, Reference_Allele, Tumor_Seq_Allele2, t_depth,
    t_alt_count, t_ref_count``.
``data_clinical_sample.txt``
    GENIE sample-level clinical table (cBioPortal format with four metadata
    header lines). Required column: ``SAMPLE_ID``. Typically also carries
    ``ONCOTREE_CODE``, ``SEQ_ASSAY_ID``, ``CANCER_TYPE``.

Both files come from the AACR Project GENIE v18.0 public release on Synapse:
https://www.synapse.org/genie

Output
------
``GENIE.csv``
    Variant table with the original MAF columns, an added ``vaf`` column,
    and the merged sample-level clinical columns.

Usage
-----
    python create_training_data.py \\
        --maf data_mutations_extended.txt \\
        --clinical data_clinical_sample.txt \\
        --output GENIE.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# Cap on Reference_Allele / Tumor_Seq_Allele2 string lengths. Long indels are
# truncated to ALLELE_FRONT_LEN leading + ALLELE_BACK_LEN trailing bases so they
# fit TESSERA's fixed-length per-token allele encoder; same convention as the
# TCGA pretraining MAF preprocessing.
ALLELE_FRONT_LEN = 5
ALLELE_BACK_LEN = 5

# cBioPortal clinical files prepend four metadata header lines (each starting
# with '#') before the column header. pandas needs to skip those to find the
# real header.
CBIOPORTAL_CLINICAL_SKIPROWS = 4

# Locus-uniqueness keys for deduplication.
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
        description="Convert GENIE MAF into the TESSERA cross-platform validation CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--maf", type=Path,
                        default=Path("data_mutations_extended.txt"),
                        help="Path to GENIE mutations MAF (tab-separated).")
    parser.add_argument("--clinical", type=Path,
                        default=Path("data_clinical_sample.txt"),
                        help="Path to GENIE sample-level clinical "
                             "table (cBioPortal format).")
    parser.add_argument("--output", type=Path, default=Path("GENIE.csv"),
                        help="Output path for the variant CSV.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.maf.exists():
        raise FileNotFoundError(f"MAF not found: {args.maf}")
    if not args.clinical.exists():
        raise FileNotFoundError(f"Clinical file not found: {args.clinical}")

    logger.info("Loading MAF: %s", args.maf)
    df_var = pd.read_csv(args.maf, sep="\t", low_memory=False)
    logger.info("MAF rows: %d", len(df_var))

    required = {"t_depth", "t_alt_count", "t_ref_count"} | set(DEDUP_KEYS)
    missing = required - set(df_var.columns)
    if missing:
        raise ValueError(f"MAF is missing required columns: {sorted(missing)}")

    n_before = len(df_var)
    df_var = df_var[(df_var["t_depth"] != 0) &
                    (df_var["t_alt_count"] >= 0) &
                    (df_var["t_ref_count"] >= 0)].copy()
    logger.info("Dropped %d rows with t_depth==0 or negative alt/ref counts (kept %d)",
                n_before - len(df_var), len(df_var))

    alt = df_var["t_alt_count"].astype(int)
    ref = df_var["t_ref_count"].astype(int)
    df_var["vaf"] = alt / (alt + ref)

    n_before = len(df_var)
    df_var = df_var.drop_duplicates(subset=DEDUP_KEYS).reset_index(drop=True)
    logger.info("Deduplicated %d -> %d rows", n_before, len(df_var))

    logger.info("Loading clinical: %s", args.clinical)
    df_clin = pd.read_csv(args.clinical, sep="\t", low_memory=False,
                          skiprows=CBIOPORTAL_CLINICAL_SKIPROWS)
    if "SAMPLE_ID" not in df_clin.columns:
        raise ValueError(
            "Clinical file must contain a SAMPLE_ID column "
            f"(found {list(df_clin.columns)[:5]}...). Confirm the file is the "
            f"cBioPortal-format sample table."
        )
    df_clin = df_clin.rename(columns={"SAMPLE_ID": "Tumor_Sample_Barcode"})

    df_out = df_var.merge(df_clin, on="Tumor_Sample_Barcode", how="left")
    if "ONCOTREE_CODE" in df_out.columns:
        n_unmapped = df_out["ONCOTREE_CODE"].isna().sum()
        if n_unmapped:
            logger.warning("%d variant rows had no ONCOTREE_CODE after clinical merge",
                           n_unmapped)

    df_out["Reference_Allele"] = df_out["Reference_Allele"].astype(str).map(truncate_allele)
    df_out["Tumor_Seq_Allele2"] = df_out["Tumor_Seq_Allele2"].astype(str).map(truncate_allele)
    df_out = df_out.reset_index(drop=True)

    logger.info("Writing %d rows to %s", len(df_out), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main(parse_args())
