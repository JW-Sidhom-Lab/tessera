"""Build the GENIE SNV table used for cross-platform validation.

Reads the merged GENIE 18.0 variant + clinical table (``GENIE.csv``,
produced by ``data/genie_18_0/create_training_data.py``), filters to
single-nucleotide variants (``Variant_Type == 'SNP'``), drops variants with
undefined VAF, subsamples up to 100 variants per sample (preserving
variants seen in at least 5 samples cohort-wide), and writes the result
as ``snv.csv``. This file is consumed by the GENIE inference scripts
that compute the SNV cross-platform-transfer accuracy reported in
Figure 1 f-g.

Inputs
------
``GENIE.csv``
    Merged GENIE variant + clinical table from the upstream Stage-1
    pipeline. Must already contain a ``vaf`` column.

Output
------
``snv.csv``
    Per-variant rows with all input columns plus ``mut_id`` (locus key
    used for the recurrent-variant subsampler).

Usage
-----
    python create_snv.py \\
        --variants ../../../data/genie_18_0/GENIE.csv \\
        --output snv.csv
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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the GENIE SNV table for cross-platform validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variants", type=Path,
                        default=Path("../../../data/genie_18_0/GENIE.csv"),
                        help="Path to the merged GENIE variant CSV.")
    parser.add_argument("--output", type=Path, default=Path("snv.csv"),
                        help="Output path for the SNV table.")
    parser.add_argument("--subsample-size", type=int, default=100,
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

    logger.info("Loading variants: %s", args.variants)
    data = pd.read_csv(args.variants, low_memory=False)
    logger.info("Raw variants: %d", len(data))

    required = {"Variant_Type", "vaf", "Tumor_Sample_Barcode"} | set(MUT_ID_COLS)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Variant CSV is missing required columns: {sorted(missing)}. "
            f"Did you run data/genie_18_0/create_training_data.py first?"
        )

    data = data[data["Variant_Type"] == "SNP"].copy()
    logger.info("After SNV filter: %d variants", len(data))

    n_before = len(data)
    data = data[data["vaf"].notna()]
    if n_before != len(data):
        logger.info("Dropped %d variants with NaN vaf (kept %d)",
                    n_before - len(data), len(data))

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

    data = data.reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(data), args.output)


if __name__ == "__main__":
    main(parse_args())
