"""Build the DepMap panel-filtered SNV, CNA, and metadata tables for cell-line validation.

Single-script pipeline: reads the DepMap 24Q2 release plus the CTRPv2
drug-response release from ``data/DepMap/``, lifts CNA segments to GRCh37,
restricts both modalities to MSK-IMPACT505 panel coverage (matching the
assay scope on which the MSK-CHORD-fit predictive biomarker $\\hat{\\tau}$
was trained), and writes the three tables consumed by the Figure 6 n
cell-line orthogonal validation:

  - ``snv_panel.csv`` -- per-variant TCGA-MAF rows on hg19, restricted to
    MSK-IMPACT505 panel genes, capped at 1,000 SNVs per cell line.
  - ``cna_panel.csv`` -- per-segment table on hg19, segments overlapping
    any IMPACT505 gene region, capped at 1,000 segments per cell line.
  - ``metadata.csv`` -- per-cell-line lineage, primary disease, subtype,
    TP53 / KRAS coding-mutation flags, 17p-loss flag, CTRPv2 oxaliplatin /
    SN-38 / fluorouracil AUCs, and modality-availability flags.

Inputs (under ``../../../data/DepMap/``)
----------------------------------------
``Model.csv``
    Cell-line metadata.
``OmicsSomaticMutations_TCGA_format_hg19.csv``
    SNVs already lifted to GRCh37 in TCGA-MAF format.
``OmicsCNSegmentsProfile.csv``
    Log2-ratio CNA segments on GRCh38; lifted here to GRCh37.
``OmicsAbsoluteCNSegmentsProfile.csv``
    Absolute-CN segments with LoH annotations; used both for the per-segment
    LOH merge and for the per-cell-line 17p-loss flag.
``OmicsDefaultModelProfiles.csv``
    ProfileID -> ModelID mapping for DNA profiles.
``CTRPv2.0_2015_ctd2_ExpandedDataset/v20.{meta.per_cell_line, meta.per_compound, meta.per_experiment, data.curves_post_qc}.txt``
    CTRPv2 drug-response area-under-curve values.

Plus the GENIE genomic-information file at
``../../../data/genie_18_0/genomic_information.txt`` for the
MSK-IMPACT505 panel gene coordinates.

Outputs (this directory)
------------------------
``snv_panel.csv``, ``cna_panel.csv``, ``metadata.csv``

Usage
-----
    python prepare_depmap_data.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

# Per-cell-line caps, matching the TCGA pretraining pipeline.
SUBSAMPLE_SNV = 1000
SUBSAMPLE_SNV_MIN_SHARED = 5
SUBSAMPLE_CNA = 1000
CNA_LOH_WEIGHT = 0.5

# Hematologic lineages excluded from the solid-tumour cohort.
HEME_LINEAGES = {"Lymphoid", "Myeloid", "Lymphoma", "Leukemia"}

# 17p hg38 boundaries used for the per-cell-line 17p-loss flag.
CHR17P_START_HG38 = 60_000
CHR17P_END_HG38 = 24_000_000

# LOH-positive tokens in OmicsAbsoluteCNSegmentsProfile.csv's LoHStatus column.
LOH_TOKENS = {
    "LOH",
    "WHOLE ARM LOH",
    "COPY-NEUTRAL LOH",
    "WHOLE ARM COPY-NEUTRAL LOH",
}

# 17p-loss thresholds: a cell line is flagged 17p-lost when its length-weighted
# absolute CN over the 17p arm divided by ploidy falls below this ratio, OR
# its LOH fraction over 17p exceeds this fraction.
RATIO_17P_LOSS_THRESHOLD = 0.85
LOH_FRAC_17P_LOSS_THRESHOLD = 0.50

# CTRPv2 drugs to merge into metadata.csv.
CTRP_DRUGS = ("oxaliplatin", "SN-38", "fluorouracil")

# Panel used to filter both modalities (matches the panel the MSK-CHORD-fit
# tau-hat was trained on).
PANEL_NAME = "MSK-IMPACT505"

# VEP-style Variant_Classification tokens treated as coding for the TP53/KRAS
# mutation flags.
CODING_VC_TOKENS = {
    "missense_variant",
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "inframe_insertion",
    "inframe_deletion",
    "start_lost",
}

# VEP-style -> TCGA Variant_Classification translation.
VEP_TO_TCGA = {
    "missense_variant": "Missense_Mutation",
    "frameshift_variant": "Frame_Shift_Del",
    "stop_gained": "Nonsense_Mutation",
    "splice_acceptor_variant": "Splice_Site",
    "splice_donor_variant": "Splice_Site",
    "splice_region_variant": "Splice_Region",
    "inframe_deletion": "In_Frame_Del",
    "inframe_insertion": "In_Frame_Ins",
    "stop_lost": "Nonstop_Mutation",
    "start_lost": "Translation_Start_Site",
    "synonymous_variant": "Silent",
    "intron_variant": "Intron",
    "5_prime_UTR_variant": "5'UTR",
    "3_prime_UTR_variant": "3'UTR",
    "non_coding_transcript_variant": "RNA",
    "non_coding_transcript_exon_variant": "RNA",
    "intergenic_variant": "IGR",
    "upstream_gene_variant": "5'Flank",
    "downstream_gene_variant": "3'Flank",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subsampling helpers (same as scripts/data/tcga/utils.py)
# ---------------------------------------------------------------------------

def _subsample_variants_with_common(data: pd.DataFrame,
                                    subsample_size: int,
                                    min_shared: int,
                                    sample_col: str = "Tumor_Sample_Barcode",
                                    random_state: int = 42) -> pd.DataFrame:
    variant_counts = data["mut_id"].value_counts()
    common_variants = set(variant_counts[variant_counts >= min_shared].index)

    def _subsample_group(group):
        common_in_sample = group[group["mut_id"].isin(common_variants)]
        remaining = subsample_size - len(common_in_sample)
        if remaining <= 0:
            return common_in_sample
        other = group[~group["mut_id"].isin(common_variants)]
        if other.empty:
            return common_in_sample
        sampled = other.sample(n=min(len(other), remaining), replace=False,
                               random_state=random_state)
        return pd.concat([common_in_sample, sampled])

    return (data.groupby(sample_col, group_keys=False)
                .apply(_subsample_group)
                .reset_index(drop=True))


def _subsample_cna_priority(data: pd.DataFrame,
                            subsample_size: int,
                            loh_weight: float,
                            sample_col: str = "Tumor_Sample_Barcode") -> pd.DataFrame:
    def _subsample_group(group):
        if len(group) <= subsample_size:
            return group
        scored = group.copy()
        scored["alteration_score"] = (
            np.abs(scored["Segment_Mean"]) +
            scored["LOH"].astype(float) * loh_weight
        )
        return scored.nlargest(subsample_size, "alteration_score").drop(columns=["alteration_score"])

    return (data.groupby(sample_col, group_keys=False)
                .apply(_subsample_group)
                .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Variant-effect / indel encoding helpers
# ---------------------------------------------------------------------------

def vep_to_tcga(vep_term: str, ref: str, alt: str) -> str:
    """Map a (possibly composite) VEP term to a TCGA Variant_Classification."""
    if not isinstance(vep_term, str) or not vep_term:
        return "Unknown"
    parts = vep_term.split("&")
    best = None
    for p in parts:
        if p in VEP_TO_TCGA and p != "splice_region_variant":
            best = VEP_TO_TCGA[p]
            break
    if best is None and parts and parts[0] in VEP_TO_TCGA:
        best = VEP_TO_TCGA[parts[0]]
    if best is None:
        return "Unknown"
    if best.startswith("Frame_Shift") or best.startswith("In_Frame"):
        ref_l = 0 if (not ref or ref == "-") else len(ref)
        alt_l = 0 if (not alt or alt == "-") else len(alt)
        if ref_l > alt_l:
            return best.replace("Ins", "Del") if "Ins" in best else best
        if alt_l > ref_l:
            return best.replace("Del", "Ins") if "Del" in best else best
    return best


def to_tcga_indel(ref: str, alt: str) -> Tuple[str, str]:
    """Convert DepMap anchor-base indel encoding to TCGA convention."""
    if not isinstance(ref, str) or not isinstance(alt, str):
        return ref, alt
    if len(ref) > len(alt):
        if ref.startswith(alt):
            return ref[len(alt):], "-"
        if ref.endswith(alt):
            return ref[:-len(alt)], "-"
    elif len(alt) > len(ref):
        if alt.startswith(ref):
            return "-", alt[len(ref):]
        if alt.endswith(ref):
            return "-", alt[:-len(ref)]
    return ref, alt


# ---------------------------------------------------------------------------
# CNA liftover
# ---------------------------------------------------------------------------

def liftover_segments(seg: pd.DataFrame) -> pd.DataFrame:
    """Lift segment coordinates from hg38 to hg19 with pyliftover.

    Drops segments where either endpoint fails to map, where strands invert,
    or where the lifted end falls before the lifted start.
    """
    from pyliftover import LiftOver
    lo = LiftOver("hg38", "hg19")

    starts, ends, drop = [], [], []
    for _, row in seg.iterrows():
        chrom = f"chr{row['Chromosome']}"
        s = lo.convert_coordinate(chrom, int(row["Start"]) - 1)
        e = lo.convert_coordinate(chrom, int(row["End"]) - 1)
        if (not s or not e
                or s[0][0] != e[0][0]
                or s[0][2] != "+" or e[0][2] != "+"):
            drop.append(True); starts.append(np.nan); ends.append(np.nan); continue
        s_pos = s[0][1] + 1
        e_pos = e[0][1] + 1
        if e_pos < s_pos:
            drop.append(True); starts.append(np.nan); ends.append(np.nan); continue
        starts.append(s_pos); ends.append(e_pos); drop.append(False)

    seg = seg.assign(Start_hg19=starts, End_hg19=ends, drop=drop)
    n_dropped = int(seg["drop"].sum())
    logger.info("Liftover dropped %d / %d segments (%.2f%%)",
                n_dropped, len(seg), 100 * n_dropped / max(1, len(seg)))
    seg = seg[~seg["drop"]].copy()
    seg["Start"] = seg["Start_hg19"].astype(int)
    seg["End"] = seg["End_hg19"].astype(int)
    return seg.drop(columns=["Start_hg19", "End_hg19", "drop"])


# ---------------------------------------------------------------------------
# Panel filtering (MSK-IMPACT505)
# ---------------------------------------------------------------------------

def load_panel_coords(panel_info_path: Path,
                      panel: str = PANEL_NAME) -> Tuple[Set[str], pd.DataFrame]:
    """Load gene set + per-gene min/max coordinate intervals for the named panel."""
    pi = pd.read_csv(panel_info_path, sep="\t", low_memory=False)
    pi = pi[pi["SEQ_ASSAY_ID"] == panel]
    pi = pi.assign(Chromosome=pi["Chromosome"].astype(str))
    genes = set(pi["Hugo_Symbol"].dropna().unique())
    regions = (pi.groupby(["Chromosome", "Hugo_Symbol"], as_index=False)
                  .agg(Start=("Start_Position", "min"),
                       End=("End_Position", "max")))
    return genes, regions


def filter_cna_to_panel(cna: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Keep CNA segments overlapping any panel gene region on the same chromosome."""
    cna = cna.assign(Chromosome=cna["Chromosome"].astype(str))
    keep = np.zeros(len(cna), dtype=bool)
    for chrom, ch_regions in regions.groupby("Chromosome"):
        idx = cna.index[cna["Chromosome"] == chrom]
        if len(idx) == 0:
            continue
        seg_starts = cna.loc[idx, "Start"].to_numpy()
        seg_ends = cna.loc[idx, "End"].to_numpy()
        r_starts = ch_regions["Start"].to_numpy()
        r_ends = ch_regions["End"].to_numpy()
        overlap = ((seg_starts[:, None] <= r_ends[None, :]) &
                   (seg_ends[:, None] >= r_starts[None, :]))
        keep[idx[overlap.any(axis=1)]] = True
    return cna.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def call_genotype_flags(metadata: pd.DataFrame,
                        snv_table: pd.DataFrame,
                        genes=("TP53", "KRAS")) -> pd.DataFrame:
    """Add coding-mutation flags to metadata (1 if cell line carries any coding
    mutation in that gene, else 0)."""
    snv = snv_table.copy()
    snv["is_coding"] = snv["VariantInfo"].apply(
        lambda v: any(t in CODING_VC_TOKENS for t in str(v).split("&"))
        if isinstance(v, str) else False
    )
    coding = snv[snv["is_coding"]]
    metadata = metadata.copy()
    for gene in genes:
        gene_samples = set(coding.loc[coding["Hugo_Symbol"] == gene,
                                       "Tumor_Sample_Barcode"])
        metadata[f"{gene}_mut"] = metadata["ModelID"].isin(gene_samples).astype(int)
    return metadata


def call_17p_loss_per_model(abs_seg_path: Path,
                            prof_to_model: Dict[str, str]) -> pd.DataFrame:
    """Compute per-cell-line 17p-loss flag from the absolute-CN segment file."""
    seg = pd.read_csv(abs_seg_path)
    seg_clean = seg.dropna(subset=["SegmentAbsoluteCN"]).copy()
    seg_clean["seg_len"] = seg_clean["End"] - seg_clean["Start"]
    seg_clean["w_cn"] = seg_clean["SegmentAbsoluteCN"] * seg_clean["seg_len"]
    ploidy = (seg_clean.groupby("ProfileID")
                  .agg(w=("seg_len", "sum"), s=("w_cn", "sum"))
                  .reset_index())
    ploidy["ploidy"] = ploidy["s"] / ploidy["w"]
    pid_ploidy = dict(zip(ploidy["ProfileID"], ploidy["ploidy"]))

    s17 = seg[seg["Chromosome"].astype(str) == "17"].copy()
    s17["ov_start"] = s17["Start"].clip(lower=CHR17P_START_HG38)
    s17["ov_end"] = s17["End"].clip(upper=CHR17P_END_HG38)
    s17["ov"] = (s17["ov_end"] - s17["ov_start"]).clip(lower=0)
    s17 = s17[s17["ov"] > 0]
    s17["loh_flag"] = s17["LoHStatus"].isin(LOH_TOKENS).astype(int)
    s17["w_cn"] = s17["SegmentAbsoluteCN"] * s17["ov"]
    s17["w_loh"] = s17["loh_flag"] * s17["ov"]
    per = (s17.dropna(subset=["SegmentAbsoluteCN"])
                .groupby("ProfileID")
                .agg(w=("ov", "sum"),
                     s_cn=("w_cn", "sum"),
                     s_loh=("w_loh", "sum"))
                .reset_index())
    per["abs_cn_17p"] = per["s_cn"] / per["w"]
    per["loh_frac_17p"] = per["s_loh"] / per["w"]
    per["ploidy"] = per["ProfileID"].map(pid_ploidy)
    per["ratio_17p"] = per["abs_cn_17p"] / per["ploidy"]
    per["has_17p_loss"] = (
        (per["ratio_17p"] < RATIO_17P_LOSS_THRESHOLD) |
        (per["loh_frac_17p"] > LOH_FRAC_17P_LOSS_THRESHOLD)
    ).astype(int)
    per["ModelID"] = per["ProfileID"].map(prof_to_model)
    per = per.dropna(subset=["ModelID"])
    return (per.groupby("ModelID")
                .agg(abs_cn_17p=("abs_cn_17p", "mean"),
                     ploidy=("ploidy", "mean"),
                     ratio_17p=("ratio_17p", "mean"),
                     loh_frac_17p=("loh_frac_17p", "mean"),
                     has_17p_loss=("has_17p_loss", "max"))
                .reset_index())


def load_ctrp_drugs(ctrp_dir: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Compute per-cell-line AUC for the CTRP_DRUGS list."""
    cell = pd.read_csv(ctrp_dir / "v20.meta.per_cell_line.txt", sep="\t")
    cpd = pd.read_csv(ctrp_dir / "v20.meta.per_compound.txt", sep="\t")
    exp = pd.read_csv(ctrp_dir / "v20.meta.per_experiment.txt", sep="\t")
    curves = pd.read_csv(ctrp_dir / "v20.data.curves_post_qc.txt", sep="\t")

    cpd_ids = {d: int(cpd.loc[cpd["cpd_name"] == d, "master_cpd_id"].iloc[0])
               for d in CTRP_DRUGS}
    e2c = dict(zip(exp["experiment_id"], exp["master_ccl_id"]))
    name_to_ccl = dict(zip(cell["ccl_name"].str.upper(), cell["master_ccl_id"]))
    curves["master_ccl_id"] = curves["experiment_id"].map(e2c)

    keep = curves[curves["master_cpd_id"].isin(cpd_ids.values())
                  & curves["master_ccl_id"].notna()].copy()
    keep["drug"] = keep["master_cpd_id"].map({v: k for k, v in cpd_ids.items()})
    agg = (keep.groupby(["master_ccl_id", "drug"], as_index=False)
                .agg(auc=("area_under_curve", "mean")))
    pivot = agg.pivot(index="master_ccl_id", columns="drug", values="auc")
    pivot.columns = [f"{c}_AUC" for c in pivot.columns]
    return pivot.reset_index(), name_to_ccl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DepMap panel-filtered SNV / CNA / metadata tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", type=Path,
                        default=Path("../../../data/DepMap"),
                        help="Directory containing the DepMap 24Q2 + CTRPv2 release files.")
    parser.add_argument("--panel-info", type=Path,
                        default=Path("../../../data/genie_18_0/genomic_information.txt"),
                        help="GENIE genomic-information file with per-panel exon coordinates.")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for snv_panel.csv, cna_panel.csv, metadata.csv.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.src.exists():
        raise FileNotFoundError(f"DepMap source directory not found: {args.src}")
    if not args.panel_info.exists():
        raise FileNotFoundError(f"Panel info not found: {args.panel_info}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- Cohort scope -----------------------------------------------------
    logger.info("Loading model metadata: %s", args.src / "Model.csv")
    m = pd.read_csv(args.src / "Model.csv")
    solid = m[~m["OncotreeLineage"].isin(HEME_LINEAGES)].copy()
    logger.info("Solid-tumour cell lines (excluding hematologic): %d", len(solid))

    prof = pd.read_csv(args.src / "OmicsDefaultModelProfiles.csv")
    dna_prof = prof[prof["ProfileType"] == "dna"]
    pid_to_model = dict(zip(dna_prof["ProfileID"], dna_prof["ModelID"]))

    # ---- Panel coordinates ------------------------------------------------
    logger.info("Loading %s panel coordinates: %s", PANEL_NAME, args.panel_info)
    panel_genes, panel_regions = load_panel_coords(args.panel_info)
    logger.info("Panel %s: %d genes, %d gene regions",
                PANEL_NAME, len(panel_genes), len(panel_regions))

    # ---- SNVs -------------------------------------------------------------
    logger.info("Loading SNVs (TCGA-format, hg19)")
    snv = pd.read_csv(args.src / "OmicsSomaticMutations_TCGA_format_hg19.csv",
                      low_memory=False)
    snv = snv[snv["Tumor_Sample_Barcode"].isin(set(solid["ModelID"]))].copy()
    logger.info("Solid-tumour SNV rows: %d (%d cell lines)",
                len(snv), snv["Tumor_Sample_Barcode"].nunique())

    # Convert indels + map VEP -> TCGA + add End_Position
    converted_ref, converted_alt = [], []
    for ref, alt in zip(snv["Reference_Allele"], snv["Tumor_Seq_Allele2"]):
        r, a = to_tcga_indel(ref, alt)
        converted_ref.append(r); converted_alt.append(a)
    snv["Reference_Allele"] = converted_ref
    snv["Tumor_Seq_Allele2"] = converted_alt
    snv["Variant_Classification"] = [
        vep_to_tcga(v, r, a)
        for v, r, a in zip(snv["VariantInfo"],
                           snv["Reference_Allele"], snv["Tumor_Seq_Allele2"])
    ]
    snv["End_Position"] = snv.apply(
        lambda r: int(r["Start_Position"]) if (not isinstance(r["Reference_Allele"], str)
                                                or r["Reference_Allele"] == "-")
                  else int(r["Start_Position"]) + len(r["Reference_Allele"]) - 1,
        axis=1,
    )
    snv["NCBI_Build"] = "GRCh37"
    snv["t_ref_count"] = snv.get("RefCount", np.nan)
    snv["t_alt_count"] = snv.get("AltCount", np.nan)
    snv["Mutation_Status"] = "SOMATIC"

    # Keep VariantInfo for downstream genotype calling, but exclude from output
    snv_for_metadata = snv[["Tumor_Sample_Barcode", "Hugo_Symbol", "VariantInfo"]].copy()

    keep_cols = [
        "Hugo_Symbol", "EntrezGeneID", "NCBI_Build", "Chromosome",
        "Start_Position", "End_Position", "Reference_Allele",
        "Tumor_Seq_Allele2", "Variant_Classification", "VariantType",
        "Tumor_Sample_Barcode", "Mutation_Status",
        "t_ref_count", "t_alt_count", "vaf", "DP",
        "DNAChange", "ProteinChange", "EnsemblFeatureID", "Hotspot",
        "VepImpact", "OncogeneHighImpact", "TumorSuppressorHighImpact",
        "LikelyLoF",
    ]
    keep_cols = [c for c in keep_cols if c in snv.columns]
    snv = snv[keep_cols]

    # Filter to single-base SNVs and per-cell-line subsample
    n_pre = len(snv)
    snv = snv[snv["VariantType"] == "SNV"].copy()
    logger.info("Filter to VariantType==SNV: %d -> %d", n_pre, len(snv))

    mut_id_cols = ["Hugo_Symbol", "Chromosome", "Start_Position",
                   "Reference_Allele", "Tumor_Seq_Allele2"]
    snv["mut_id"] = snv[mut_id_cols].astype(str).agg("_".join, axis=1)
    snv = _subsample_variants_with_common(
        snv, subsample_size=SUBSAMPLE_SNV,
        min_shared=SUBSAMPLE_SNV_MIN_SHARED,
        sample_col="Tumor_Sample_Barcode",
    )
    # mut_id (the subsampler's locus key) is retained in the output for
    # schema consistency with the original manuscript snv_panel.csv.
    logger.info("After SNV subsample: %d rows on %d cell lines",
                len(snv), snv["Tumor_Sample_Barcode"].nunique())

    n_pre = len(snv)
    snv_panel = snv[snv["Hugo_Symbol"].isin(panel_genes)].reset_index(drop=True)
    logger.info("After %s panel filter: %d -> %d rows on %d cell lines",
                PANEL_NAME, n_pre, len(snv_panel),
                snv_panel["Tumor_Sample_Barcode"].nunique())
    snv_panel.to_csv(out / "snv_panel.csv", index=False)
    logger.info("Wrote %s", out / "snv_panel.csv")

    # ---- CNA --------------------------------------------------------------
    logger.info("Loading CNA segments (hg38) and lifting to hg19")
    cn = pd.read_csv(args.src / "OmicsCNSegmentsProfile.csv")
    cn = cn[cn["ProfileID"].isin(set(dna_prof["ProfileID"]))].copy()
    cn["ModelID"] = cn["ProfileID"].map(pid_to_model)
    cn = cn[cn["ModelID"].isin(set(solid["ModelID"]))].copy()
    logger.info("Pre-liftover segments: %d (%d cell lines)",
                len(cn), cn["ModelID"].nunique())

    abs_seg = pd.read_csv(args.src / "OmicsAbsoluteCNSegmentsProfile.csv",
                          usecols=["ProfileID", "Chromosome", "Start", "End",
                                   "LoHStatus"])
    abs_seg["LOH"] = abs_seg["LoHStatus"].isin(LOH_TOKENS).astype(int)
    cn["Chromosome"] = cn["Chromosome"].astype(str)
    abs_seg["Chromosome"] = abs_seg["Chromosome"].astype(str)
    cn = cn.merge(
        abs_seg[["ProfileID", "Chromosome", "Start", "End", "LOH"]],
        on=["ProfileID", "Chromosome", "Start", "End"], how="left",
    )
    cn["LOH"] = cn["LOH"].fillna(0).astype(int)
    logger.info("LOH-positive segments: %d / %d (%.1f%%)",
                int(cn["LOH"].sum()), len(cn), cn["LOH"].mean() * 100)

    cn = liftover_segments(cn)
    cn = (cn.rename(columns={"ModelID": "Tumor_Sample_Barcode",
                             "SegmentMean": "Segment_Mean"})
              [["Tumor_Sample_Barcode", "Chromosome", "Start", "End",
                "Segment_Mean", "NumProbes", "LOH"]])

    n_pre = len(cn)
    cn = _subsample_cna_priority(
        cn, subsample_size=SUBSAMPLE_CNA, loh_weight=CNA_LOH_WEIGHT,
        sample_col="Tumor_Sample_Barcode",
    )
    logger.info("After CNA subsample: %d -> %d segments on %d cell lines",
                n_pre, len(cn), cn["Tumor_Sample_Barcode"].nunique())

    n_pre = len(cn)
    cna_panel = filter_cna_to_panel(cn, panel_regions)
    logger.info("After %s panel filter: %d -> %d segments on %d cell lines",
                PANEL_NAME, n_pre, len(cna_panel),
                cna_panel["Tumor_Sample_Barcode"].nunique())
    cna_panel.to_csv(out / "cna_panel.csv", index=False)
    logger.info("Wrote %s", out / "cna_panel.csv")

    # ---- Metadata ---------------------------------------------------------
    logger.info("Building per-cell-line metadata")
    meta = solid[["ModelID", "StrippedCellLineName",
                  "OncotreeLineage", "OncotreePrimaryDisease",
                  "OncotreeSubtype"]].copy()
    meta = call_genotype_flags(meta, snv_for_metadata)

    cn17 = call_17p_loss_per_model(
        args.src / "OmicsAbsoluteCNSegmentsProfile.csv", pid_to_model,
    )
    meta = meta.merge(cn17, on="ModelID", how="left")
    meta["has_17p_loss"] = meta["has_17p_loss"].fillna(-1).astype(int)

    drug, name_to_ccl = load_ctrp_drugs(args.src / "CTRPv2.0_2015_ctd2_ExpandedDataset")
    meta["_upper"] = meta["StrippedCellLineName"].str.upper()
    meta["master_ccl_id"] = meta["_upper"].map(name_to_ccl)
    meta = meta.merge(drug, on="master_ccl_id", how="left")
    meta = meta.drop(columns=["_upper"])

    samples_with_snv = set(snv_panel["Tumor_Sample_Barcode"])
    samples_with_cna = set(cna_panel["Tumor_Sample_Barcode"])
    full = samples_with_snv & samples_with_cna
    meta["has_snv"] = meta["ModelID"].isin(samples_with_snv).astype(int)
    meta["has_cna"] = meta["ModelID"].isin(samples_with_cna).astype(int)
    meta["has_both"] = meta["ModelID"].isin(full).astype(int)
    meta.to_csv(out / "metadata.csv", index=False)
    logger.info("Wrote %s (%d cell lines)", out / "metadata.csv", len(meta))

    logger.info("Coverage: SNV=%d cell lines, CNA=%d, both=%d",
                len(samples_with_snv), len(samples_with_cna), len(full))


if __name__ == "__main__":
    main(parse_args())
