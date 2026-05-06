"""Convert the ClinVar VCF release into the pickled DataFrame used for variant-pathogenicity prediction.

Reads a NCBI ClinVar VCF (compressed or uncompressed), parses three fields out
of the semicolon-delimited ``INFO`` column (``ALLELEID``, ``CLNSIG``,
``GENEINFO``), drops the original ``INFO`` and any intermediate parse
artefacts, and pickles the cleaned DataFrame. Downstream variant-effect
scripts join this table to the model's variant predictions to evaluate
pathogenicity classification (Fig.~1h--o in the manuscript).

Input
-----
``clinvar_<release-date>.vcf.gz`` (or ``.vcf``)
    NCBI ClinVar VCF release, GRCh37 build. Pandas reads the gzipped form
    directly via ``compression='infer'``. Source:
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/

Output
------
``clinvar.pkl``
    Pickled pandas DataFrame with columns ``CHROM, POS, ID, REF, ALT, QUAL,
    FILTER, ALLELEID, CLNSIG, GENEINFO``. Each row is one variant; ``CLNSIG``
    is the pathogenicity classification used as the label downstream.

Usage
-----
    python create_training_data.py \\
        --vcf clinvar_20241230.vcf.gz \\
        --output clinvar.pkl
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# VCF column header for the eight standard fields (per VCF spec). ClinVar files
# omit per-sample columns, so this is the complete column list.
VCF_COLUMNS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]

# INFO subfields parsed out and promoted to top-level columns. ClinVar carries
# many more INFO keys (CLNREVSTAT, CLNDN, CLNVC, etc.); only these three are
# consumed by the downstream variant-effect prediction scripts.
INFO_FIELDS = ["ALLELEID", "CLNSIG", "GENEINFO"]

logger = logging.getLogger(__name__)


def parse_info(info: str) -> dict:
    """Split a VCF ``INFO`` string into a dict of {key: value} entries."""
    out = {}
    for item in info.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
        else:
            # Flag-style entries (no value) are recorded as None.
            out[item] = None
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse the ClinVar VCF into a pickled DataFrame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vcf", type=Path,
                        default=Path("clinvar_20241230.vcf.gz"),
                        help="Path to the ClinVar VCF (.vcf or .vcf.gz).")
    parser.add_argument("--output", type=Path, default=Path("clinvar.pkl"),
                        help="Output path for the pickled DataFrame.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.vcf.exists():
        raise FileNotFoundError(f"VCF not found: {args.vcf}")

    logger.info("Loading VCF: %s", args.vcf)
    df = pd.read_csv(
        args.vcf,
        comment="#",
        sep="\t",
        names=VCF_COLUMNS,
        dtype=str,
        compression="infer",
    )
    logger.info("VCF rows: %d", len(df))

    parsed = df["INFO"].map(parse_info)
    for field in INFO_FIELDS:
        df[field] = parsed.map(lambda d, f=field: d.get(f))

    n_with_clnsig = df["CLNSIG"].notna().sum()
    logger.info("Variants with CLNSIG: %d (%.1f%%)",
                n_with_clnsig, 100 * n_with_clnsig / len(df))

    df = df.drop(columns=["INFO"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d rows to %s", len(df), args.output)
    df.to_pickle(args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main(parse_args())
