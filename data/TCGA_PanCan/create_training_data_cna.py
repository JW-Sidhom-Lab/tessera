"""Convert TCGA ABSOLUTE copy-number segments into the CNA CSV used for TESSERA pretraining.

Reads the TCGA Pan-Cancer ABSOLUTE allele-specific copy-number segments and
the matched purity/ploidy table, computes a purity-adjusted log2 segment-mean
for each segment, and writes the result as a comma-separated table.
Downstream pretraining scripts split this table into train/valid partitions,
subsample to a fixed budget per sample by alteration magnitude, and feed
``Segment_Mean`` and ``LOH`` to the CNA encoder as ``cna_segment_mean`` and
``cna_loh``.

Inputs
------
``TCGA_mastercalls.abs_segtabs.fixed.txt``
    ABSOLUTE allele-specific copy-number segments (Carter et al., Nat. Biotech.
    2012). Tab-separated; one row per segment per sample. Required columns:
    ``Sample, Chromosome, Start, End, Modal_Total_CN, LOH``.
``TCGA_mastercalls.abs_tables_JSedit.fixed.txt``
    ABSOLUTE per-sample purity and ploidy estimates. Tab-separated. Required
    columns: ``array, sample, purity, ploidy``.

Both files are distributed together on the TCGA Pan-Cancer Atlas downloads
page: https://gdc.cancer.gov/about-data/publications/pancanatlas

Output
------
``TCGA_CNA.csv``
    Segment table with columns ``Sample, Chromosome, Start, End, Segment_Mean,
    LOH``. ``Segment_Mean`` is the purity-adjusted log2 copy-ratio relative to
    a diploid baseline.

Usage
-----
    python create_training_data_cna.py \\
        --segments TCGA_mastercalls.abs_segtabs.fixed.txt \\
        --purity TCGA_mastercalls.abs_tables_JSedit.fixed.txt \\
        --output TCGA_CNA.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Default purity for samples missing an ABSOLUTE estimate. ABSOLUTE solutions
# fail on a small minority of samples; assigning 0.8 (a typical solid-tumor
# purity) lets downstream Segment_Mean values remain meaningful for those
# samples without dropping them from pretraining.
DEFAULT_PURITY = 0.8

# Diploid baseline copy number used when normalising to log2 ratios.
DIPLOID_CN = 2

# Columns retained from the segments file. Modal_Total_CN is required to
# compute Segment_Mean and is dropped before output.
SEG_COLS = ["Sample", "Chromosome", "Start", "End", "Modal_Total_CN", "LOH"]

# Final output columns.
OUTPUT_COLS = ["Sample", "Chromosome", "Start", "End", "Segment_Mean", "LOH"]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TCGA ABSOLUTE segments into the TESSERA CNA CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--segments", type=Path,
                        default=Path("TCGA_mastercalls.abs_segtabs.fixed.txt"),
                        help="Path to ABSOLUTE allele-specific segments (tab-separated).")
    parser.add_argument("--purity", type=Path,
                        default=Path("TCGA_mastercalls.abs_tables_JSedit.fixed.txt"),
                        help="Path to ABSOLUTE purity/ploidy table (tab-separated).")
    parser.add_argument("--output", type=Path, default=Path("TCGA_CNA.csv"),
                        help="Output path for the CNA segment CSV.")
    return parser.parse_args()


def load_segments(path: Path) -> pd.DataFrame:
    """Load ABSOLUTE segments and return a typed, NaN-clean frame."""
    logger.info("Loading segments: %s", path)
    df = pd.read_csv(path, sep="\t")
    logger.info("Segments loaded: %d rows from %d samples",
                len(df), df["Sample"].nunique())

    missing = set(SEG_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Segments file is missing required columns: {sorted(missing)}")

    df = df[SEG_COLS].copy()
    n_before = len(df)
    df = df.dropna(subset=["Chromosome", "Start", "End", "Modal_Total_CN"])
    logger.info("Dropped %d rows with NaN in coordinate/CN columns",
                n_before - len(df))

    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    df["Modal_Total_CN"] = df["Modal_Total_CN"].astype(int)
    return df


def load_purity(path: Path) -> pd.DataFrame:
    """Load ABSOLUTE purity/ploidy table; return columns ``Sample, purity``."""
    logger.info("Loading purity table: %s", path)
    df = pd.read_csv(path, sep="\t")

    required = {"array", "sample", "purity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Purity file is missing required columns: {sorted(missing)}")

    df = df[["array", "purity"]].rename(columns={"array": "Sample"})
    logger.info("Purity loaded: %d samples", len(df))
    return df


def compute_segment_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Add a purity-adjusted log2 copy-ratio column ``Segment_Mean`` in-place."""
    # Tumor-purity correction: an observed bulk-sequencing CN is a mixture of
    # the tumor CN and the (assumed diploid) normal contamination. Inverting
    # that mixture before taking the log2 ratio gives a Segment_Mean comparable
    # across samples of different purities.
    observed_cn = df["purity"] * df["Modal_Total_CN"] + (1 - df["purity"]) * DIPLOID_CN
    df["Segment_Mean"] = np.log2(observed_cn / DIPLOID_CN)
    df["Segment_Mean"] = df["Segment_Mean"].replace([np.inf, -np.inf], np.nan)
    logger.info("Segment_Mean: mean=%.3f median=%.3f",
                df["Segment_Mean"].mean(), df["Segment_Mean"].median())
    return df


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.segments.exists():
        raise FileNotFoundError(f"Segments file not found: {args.segments}")
    if not args.purity.exists():
        raise FileNotFoundError(f"Purity file not found: {args.purity}")

    df_seg = load_segments(args.segments)
    df_purity = load_purity(args.purity)

    df = df_seg.merge(df_purity, on="Sample", how="left")
    n_missing_purity = df["purity"].isna().sum()
    if n_missing_purity:
        logger.warning("Filling %d segments missing purity with default %.2f",
                       n_missing_purity, DEFAULT_PURITY)
        df["purity"] = df["purity"].fillna(DEFAULT_PURITY)

    df = compute_segment_mean(df)

    df["LOH"] = df["LOH"].astype(bool)
    df["Chromosome"] = df["Chromosome"].astype(int).astype(str)

    n_before = len(df)
    df = df.dropna(subset=["Segment_Mean"])
    if n_before != len(df):
        logger.info("Dropped %d rows with NaN Segment_Mean", n_before - len(df))

    df_out = df[OUTPUT_COLS].reset_index(drop=True)
    logger.info("Writing %d segments from %d samples to %s",
                len(df_out), df_out["Sample"].nunique(), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main(parse_args())
