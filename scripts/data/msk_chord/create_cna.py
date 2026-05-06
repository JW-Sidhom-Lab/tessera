"""Build the MSK-CHORD CNA tables used for cross-platform validation and clinical analyses.

Reads the MSK-CHORD CNA segments file and the sample-to-panel mapping,
restricts to solid-tumor samples (excluding the IMPACT-HEME-400 panel),
and writes two outputs:

1. ``cna.csv`` -- raw CNA segments for solid-tumor samples in the four
   IMPACT panel versions (341, 410, 468, 505), with TCGA-style column
   names. This is the table used for the CNA cross-platform-validation
   evaluation in Figure 2 d.

2. ``cna_panel_filtered.csv`` -- the same segments trimmed to the gene
   regions covered by each sample's specific IMPACT panel, using gene
   coordinates from the GENIE 18.0 ``genomic_information.txt`` file.
   This is the table used for the joint-modality predictive-biomarker
   analyses in Figure 6.

Inputs
------
``data_cna_hg19.seg``
    MSK-CHORD CNA segments (cBioPortal SEG format).
``data_gene_panel_matrix.txt``
    Sample-to-panel mapping (column ``cna`` carries the panel name).
``genomic_information.txt``
    GENIE genomic-information file giving per-panel gene/exon coordinates;
    we aggregate to gene-level (one min/max-bounded region per gene).
    The four MSK-IMPACT panels we filter to are stable across GENIE
    releases, so the choice of release version does not affect outputs.

Outputs
-------
``cna.csv``
    Columns: ``Tumor_Sample_Barcode, Chromosome, Start, End, Segment_Mean``.
``cna_panel_filtered.csv``
    Same columns plus ``Hugo_Symbol`` and ``Panel``; segment coordinates
    trimmed to the overlap with each gene region.

Usage
-----
    python create_cna.py \\
        --segments ../../../data/msk_chord_2024/data_cna_hg19.seg \\
        --sample-panel ../../../data/msk_chord_2024/data_gene_panel_matrix.txt \\
        --panel-info ../../../data/genie_18_0/genomic_information.txt \\
        --output-raw cna.csv \\
        --output-filtered cna_panel_filtered.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Solid-tumor IMPACT panels (excludes IMPACT-HEME-400 hematologic panel).
INCLUDE_PANELS = ["IMPACT341", "IMPACT410", "IMPACT468", "IMPACT505"]

# Map MSK-CHORD panel labels to GENIE genomic-information ``SEQ_ASSAY_ID`` keys.
PANEL_NAME_MAP = {
    "IMPACT341": "MSK-IMPACT341",
    "IMPACT410": "MSK-IMPACT410",
    "IMPACT468": "MSK-IMPACT468",
    "IMPACT505": "MSK-IMPACT505",
}

# Chunk size for the cross-merge inside the per-chromosome panel filter; tuned
# to keep peak memory within ~4 GB on the largest chromosomes.
PANEL_FILTER_CHUNK_SIZE = 50_000

# CNA segment column rename (cBioPortal SEG to TCGA convention).
SEG_COLUMN_RENAME = {
    "ID": "Tumor_Sample_Barcode",
    "chrom": "Chromosome",
    "loc.start": "Start",
    "loc.end": "End",
    "seg.mean": "Segment_Mean",
}

logger = logging.getLogger(__name__)


def load_sample_panel_mapping(path: Path) -> Dict[str, str]:
    """Return a sample_id -> panel_name dict from the cBioPortal panel matrix."""
    logger.info("Loading sample-panel mapping: %s", path)
    df = pd.read_csv(path, sep="\t")
    sample_to_panel = dict(zip(df["SAMPLE_ID"], df["cna"]))
    logger.info("Total samples: %d", len(sample_to_panel))
    for panel, count in df["cna"].value_counts().items():
        logger.info("  %s: %d", panel, count)
    return sample_to_panel


def load_panel_coordinates(path: Path,
                           panel_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Return per-panel gene-region coordinate frames from GENIE genomic-information.

    For each panel we aggregate to one (Chromosome, Start, End) interval per
    gene, taking the min Start and max End across exons.
    """
    logger.info("Loading panel coordinates: %s", path)
    panel_df = pd.read_csv(path, sep="\t", low_memory=False)
    panel_df["Chromosome"] = panel_df["Chromosome"].astype(str)

    coords_by_panel: Dict[str, pd.DataFrame] = {}
    for panel_name in panel_names:
        genie_name = PANEL_NAME_MAP.get(panel_name, panel_name)
        coords = panel_df[panel_df["SEQ_ASSAY_ID"] == genie_name]
        if coords.empty:
            logger.warning("No coordinates for panel %s", genie_name)
            continue
        coords = (coords.rename(columns={"Start_Position": "Start", "End_Position": "End"})
                        [["Chromosome", "Start", "End", "Hugo_Symbol"]]
                        .groupby(["Chromosome", "Hugo_Symbol"], as_index=False)
                        .agg(Start=("Start", "min"), End=("End", "max")))
        coords_by_panel[panel_name] = coords
        logger.info("  %s: %d gene regions", panel_name, len(coords))
    return coords_by_panel


def load_cna_segments(path: Path) -> pd.DataFrame:
    """Load the cBioPortal SEG-format CNA segments and rename to TCGA columns."""
    logger.info("Loading CNA segments: %s", path)
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns=SEG_COLUMN_RENAME)
    df["Chromosome"] = df["Chromosome"].astype(str)
    logger.info("Loaded %d segments from %d samples",
                len(df), df["Tumor_Sample_Barcode"].nunique())
    return df


def filter_to_panel(cna_df: pd.DataFrame,
                    panel_df: pd.DataFrame) -> pd.DataFrame:
    """Trim CNA segments to overlap with a single panel's gene regions.

    Per-chromosome cross-merge of CNA segments against the panel's gene
    intervals, keeping only overlapping pairs and trimming each segment to
    the intersection with the gene region. Returns a frame with one row per
    (segment, overlapping gene) pair.
    """
    cna_chroms = set(cna_df["Chromosome"])
    panel_chroms = set(panel_df["Chromosome"])
    common_chroms = cna_chroms & panel_chroms

    pieces: List[pd.DataFrame] = []
    for chrom in sorted(common_chroms, key=lambda x: (len(x), x)):
        cna_chr = cna_df[cna_df["Chromosome"] == chrom]
        panel_chr = panel_df[panel_df["Chromosome"] == chrom]
        if cna_chr.empty or panel_chr.empty:
            continue

        panel_chr = (panel_chr.drop(columns=["Chromosome"])
                              .rename(columns={"Start": "Panel_Start",
                                               "End": "Panel_End"}))

        for chunk_start in range(0, len(cna_chr), PANEL_FILTER_CHUNK_SIZE):
            cna_chunk = cna_chr.iloc[chunk_start:chunk_start + PANEL_FILTER_CHUNK_SIZE]
            merged = cna_chunk.merge(panel_chr, how="cross")
            overlaps = merged[(merged["Start"] < merged["Panel_End"]) &
                              (merged["End"] > merged["Panel_Start"])].copy()
            if overlaps.empty:
                continue
            overlaps["Trimmed_Start"] = np.maximum(overlaps["Start"], overlaps["Panel_Start"])
            overlaps["Trimmed_End"] = np.minimum(overlaps["End"], overlaps["Panel_End"])
            pieces.append(overlaps[[
                "Tumor_Sample_Barcode", "Chromosome",
                "Trimmed_Start", "Trimmed_End",
                "Segment_Mean", "Hugo_Symbol",
            ]].rename(columns={"Trimmed_Start": "Start", "Trimmed_End": "End"}))

    if not pieces:
        return pd.DataFrame(columns=[
            "Tumor_Sample_Barcode", "Chromosome", "Start", "End",
            "Segment_Mean", "Hugo_Symbol",
        ])
    out = pd.concat(pieces, ignore_index=True)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    return out


def filter_segments_per_sample_panel(cna_df: pd.DataFrame,
                                     sample_to_panel: Dict[str, str],
                                     panel_coords: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Apply per-sample panel filtering across all four IMPACT panels."""
    panel_to_samples: Dict[str, List[str]] = {}
    for sample, panel in sample_to_panel.items():
        panel_to_samples.setdefault(panel, []).append(sample)

    pieces: List[pd.DataFrame] = []
    for panel_name in INCLUDE_PANELS:
        if panel_name not in panel_to_samples:
            continue
        samples = panel_to_samples[panel_name]
        coords = panel_coords.get(panel_name)
        if coords is None:
            logger.warning("No coordinates for %s; skipping %d samples",
                           panel_name, len(samples))
            continue

        panel_cna = cna_df[cna_df["Tumor_Sample_Barcode"].isin(samples)]
        if panel_cna.empty:
            continue
        logger.info("Filtering %s: %d samples, %d segments",
                    panel_name, len(samples), len(panel_cna))

        filtered = filter_to_panel(panel_cna, coords)
        if not filtered.empty:
            filtered["Panel"] = panel_name
            pieces.append(filtered)

    if not pieces:
        return pd.DataFrame(columns=[
            "Tumor_Sample_Barcode", "Chromosome", "Start", "End",
            "Segment_Mean", "Hugo_Symbol", "Panel",
        ])
    return pd.concat(pieces, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw and panel-filtered MSK-CHORD CNA tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--segments", type=Path,
                        default=Path("../../../data/msk_chord_2024/data_cna_hg19.seg"),
                        help="MSK-CHORD CNA segments (cBioPortal SEG).")
    parser.add_argument("--sample-panel", type=Path,
                        default=Path("../../../data/msk_chord_2024/data_gene_panel_matrix.txt"),
                        help="MSK-CHORD sample-to-panel mapping.")
    parser.add_argument("--panel-info", type=Path,
                        default=Path("../../../data/genie_18_0/genomic_information.txt"),
                        help="GENIE genomic-information file with per-panel gene coordinates.")
    parser.add_argument("--output-raw", type=Path, default=Path("cna.csv"),
                        help="Output path for the raw solid-tumor CNA segments.")
    parser.add_argument("--output-filtered", type=Path,
                        default=Path("cna_panel_filtered.csv"),
                        help="Output path for the panel-trimmed CNA segments.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    for path in (args.segments, args.sample_panel, args.panel_info):
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

    sample_to_panel = load_sample_panel_mapping(args.sample_panel)
    solid_tumor_samples = {s for s, p in sample_to_panel.items() if p in INCLUDE_PANELS}
    logger.info("Solid-tumor samples (excluding HEME): %d", len(solid_tumor_samples))

    cna_df = load_cna_segments(args.segments)
    cna_df = cna_df[cna_df["Tumor_Sample_Barcode"].isin(solid_tumor_samples)]
    logger.info("After solid-tumor filter: %d segments from %d samples",
                len(cna_df), cna_df["Tumor_Sample_Barcode"].nunique())

    cna_raw = cna_df[["Tumor_Sample_Barcode", "Chromosome",
                      "Start", "End", "Segment_Mean"]]
    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    cna_raw.to_csv(args.output_raw, index=False)
    logger.info("Wrote %d raw segments -> %s", len(cna_raw), args.output_raw)

    panel_coords = load_panel_coordinates(args.panel_info, INCLUDE_PANELS)
    cna_filtered = filter_segments_per_sample_panel(
        cna_raw, sample_to_panel, panel_coords,
    )
    args.output_filtered.parent.mkdir(parents=True, exist_ok=True)
    cna_filtered.to_csv(args.output_filtered, index=False)
    logger.info("Wrote %d panel-filtered segments -> %s",
                len(cna_filtered), args.output_filtered)


if __name__ == "__main__":
    main(parse_args())
