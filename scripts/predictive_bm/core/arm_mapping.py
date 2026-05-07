"""Chromosomal-arm mapping for CNA attribution.

Why: panel-level CN segments are typically arm-level or larger. Per-gene
aggregation (in `gene_mapping.py`) attributes each segment equally across
all panel genes it overlaps, which is statistically faithful to the panel
data resolution but biologically misleading - a chr17p loss reads as
"contributions from TP53, MAP2K4, NF1, RPA1, ..." instead of "chr17p loss".

This module aggregates CNA attribution by chromosomal arm instead, matching
the resolution of clinical/biological interpretation. SNVs continue to use
gene-level aggregation (where the locus IS the relevant biology).

Reference: hg19 (GRCh37) centromere midpoints and chromosome lengths from
UCSC cytoBand.txt. Acrocentric short arms (13p, 14p, 15p, 21p, 22p) are
included for completeness but rarely contain panel targets so will carry
little to no attribution mass.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd


# hg19 centromere midpoint (the boundary between p and q arms).
# Sourced from UCSC cytoBand.txt acen entries (rounded to nearest Mb).
HG19_CENTROMERE_MIDPOINT = {
    "1":  125_000_000,
    "2":   93_300_000,
    "3":   91_000_000,
    "4":   50_400_000,
    "5":   48_400_000,
    "6":   61_000_000,
    "7":   59_900_000,
    "8":   45_600_000,
    "9":   49_000_000,
    "10":  40_200_000,
    "11":  53_700_000,
    "12":  35_800_000,
    "13":  17_900_000,    # acrocentric
    "14":  17_600_000,    # acrocentric
    "15":  19_000_000,    # acrocentric
    "16":  36_800_000,
    "17":  25_100_000,
    "18":  17_200_000,
    "19":  28_500_000,
    "20":  27_500_000,
    "21":  13_200_000,    # acrocentric
    "22":  14_700_000,    # acrocentric
    "X":   60_600_000,
    "Y":   12_500_000,
}

HG19_CHROM_LENGTH = {
    "1":  249_250_621,
    "2":  243_199_373,
    "3":  198_022_430,
    "4":  191_154_276,
    "5":  180_915_260,
    "6":  171_115_067,
    "7":  159_138_663,
    "8":  146_364_022,
    "9":  141_213_431,
    "10": 135_534_747,
    "11": 135_006_516,
    "12": 133_851_895,
    "13": 115_169_878,
    "14": 107_349_540,
    "15": 102_531_392,
    "16":  90_354_753,
    "17":  81_195_210,
    "18":  78_077_248,
    "19":  59_128_983,
    "20":  63_025_520,
    "21":  48_129_895,
    "22":  51_304_566,
    "X":  155_270_560,
    "Y":   59_373_566,
}


def build_arm_coords() -> pd.DataFrame:
    """Build a DataFrame of chromosomal arm intervals (hg19).

    Returns: arm | chr | start | end (one row per arm, 48 rows for 24 chromosomes
    each with p and q).
    """
    rows = []
    for chrom in HG19_CHROM_LENGTH:
        cen = HG19_CENTROMERE_MIDPOINT[chrom]
        length = HG19_CHROM_LENGTH[chrom]
        rows.append({"arm": f"{chrom}p", "chr": chrom, "start": 0,   "end": cen})
        rows.append({"arm": f"{chrom}q", "chr": chrom, "start": cen, "end": length})
    return pd.DataFrame(rows)


def overlap_segments_with_arms(cna_meta: pd.DataFrame) -> pd.DataFrame:
    """Map each CNA segment to its overlapping arm(s), with overlap length.

    cna_meta must have columns: row_id, Chromosome, Start, End.

    Returns long-format DataFrame: row_id | arm | overlap_bp.
    A segment that spans the centromere appears in two rows (one per arm)
    with the corresponding overlap_bp. A segment fully within a single arm
    appears in one row with overlap_bp = segment length.
    """
    arms = build_arm_coords()
    cna = cna_meta[["row_id", "Chromosome", "Start", "End"]].copy()
    cna["Chromosome"] = cna["Chromosome"].astype(str)

    out_rows = []
    for chrom, segs in cna.groupby("Chromosome"):
        if chrom not in HG19_CHROM_LENGTH:
            # Unknown chromosome (mitochondrial, alt contigs); skip
            continue
        arm_p = arms[(arms["chr"] == chrom) & arms["arm"].str.endswith("p")].iloc[0]
        arm_q = arms[(arms["chr"] == chrom) & arms["arm"].str.endswith("q")].iloc[0]
        cen = arm_p["end"]   # = arm_q["start"]
        for ridx, s_start, s_end in zip(segs["row_id"].values,
                                          segs["Start"].values,
                                          segs["End"].values):
            # p-arm overlap
            p_start = max(int(s_start), int(arm_p["start"]))
            p_end   = min(int(s_end),   int(arm_p["end"]))
            p_ov    = max(0, p_end - p_start)
            # q-arm overlap
            q_start = max(int(s_start), int(arm_q["start"]))
            q_end   = min(int(s_end),   int(arm_q["end"]))
            q_ov    = max(0, q_end - q_start)
            if p_ov > 0:
                out_rows.append((int(ridx), arm_p["arm"], p_ov))
            if q_ov > 0:
                out_rows.append((int(ridx), arm_q["arm"], q_ov))

    return pd.DataFrame(out_rows, columns=["row_id", "arm", "overlap_bp"])


def attach_arms_to_attributions(
    var_attr: pd.DataFrame,
    snv_meta: pd.DataFrame,
    cna_meta: pd.DataFrame,
    cna_to_arm: pd.DataFrame,
) -> pd.DataFrame:
    """Add a `feature` column to per-variant attributions.

    For SNV rows: feature = snv_meta.loc[row_id, 'Hugo_Symbol'] (gene name).
    For CNA rows: each row_id may map to 1 or 2 arms via cna_to_arm; we
                  duplicate the row, splitting attribution proportional to
                  overlap_bp.

    Returns long-format DataFrame: pid | source | row_id | feature | attribution
    where `feature` is a gene name (for SNVs) or an arm label (for CNAs;
    e.g. "17p", "8q").
    """
    # SNV: direct lookup of Hugo_Symbol
    snv_rows = var_attr[var_attr["source"] == "snv"].copy()
    snv_rows["feature"] = snv_rows["row_id"].map(snv_meta["Hugo_Symbol"]).values

    # CNA: expand via cna_to_arm with proportional split by overlap_bp
    cna_rows = var_attr[var_attr["source"] == "cna"].copy()
    if len(cna_rows) > 0 and len(cna_to_arm) > 0:
        # Total overlap_bp per segment (sum across arms it overlaps)
        total_ov = (cna_to_arm.groupby("row_id")["overlap_bp"].sum()
                                   .rename("total_overlap_bp").reset_index())
        cna_long = cna_rows.merge(cna_to_arm[["row_id", "arm", "overlap_bp"]],
                                       on="row_id", how="inner")
        cna_long = cna_long.merge(total_ov, on="row_id", how="left")
        cna_long["attribution"] = (cna_long["attribution"]
                                       * cna_long["overlap_bp"]
                                       / cna_long["total_overlap_bp"])
        cna_long = cna_long.rename(columns={"arm": "feature"})
        cna_long = cna_long[["pid", "source", "row_id", "feature", "attribution"]]
    else:
        cna_long = pd.DataFrame(columns=["pid", "source", "row_id",
                                            "feature", "attribution"])

    out = pd.concat([
        snv_rows[["pid", "source", "row_id", "feature", "attribution"]],
        cna_long,
    ], ignore_index=True)
    return out


def feature_attribution_per_patient(attr_with_feature: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-(patient, feature) sums.

    Returns: pid | feature | attribution | n_alterations | n_snv | n_cna
    """
    g = attr_with_feature.dropna(subset=["feature"]).copy()
    g["is_snv"] = (g["source"] == "snv").astype(int)
    g["is_cna"] = (g["source"] == "cna").astype(int)
    out = (g.groupby(["pid", "feature"])
              .agg(attribution=("attribution", "sum"),
                     n_alterations=("attribution", "size"),
                     n_snv=("is_snv", "sum"),
                     n_cna=("is_cna", "sum"))
              .reset_index())
    return out


def compute_arm_direction_map(
    A: np.ndarray, features: list[str], is_arm: np.ndarray,
    cna_meta: pd.DataFrame, patients: list[str],
) -> dict[str, dict]:
    """For each arm column in A, compute the Spearman correlation between
    the column's attribution values and the patient's mean log2 ratio at
    that arm. The sign of this correlation tells us whether high attribution
    on this arm corresponds to AMP (rho > 0) or LOSS (rho < 0).

    Returns: dict mapping arm_label -> {"rho": float, "P": float, "sign": "AMP" | "LOSS"}.
    Used downstream to label signature loadings biologically.
    """
    from scipy.stats import spearmanr
    cna = cna_meta.copy()
    cna["pid"] = cna["Tumor_Sample_Barcode"].str[:9] if "pid" not in cna.columns else cna["pid"]
    chrom_to_arms = build_arm_coords()
    chrom_to_arms = {c: chrom_to_arms[chrom_to_arms["chr"] == c]
                       for c in chrom_to_arms["chr"].unique()}

    # length-weighted per-(pid, arm) mean log2
    rows = []
    for _, seg in cna[["pid", "Chromosome", "Start", "End", "Segment_Mean"]].iterrows():
        chrom = str(seg["Chromosome"])
        if chrom not in chrom_to_arms:
            continue
        for _, arm in chrom_to_arms[chrom].iterrows():
            ov = max(0, min(int(seg["End"]), int(arm["end"]))
                        - max(int(seg["Start"]), int(arm["start"])))
            if ov > 0:
                rows.append((seg["pid"], arm["arm"], seg["Segment_Mean"], ov))
    raw = pd.DataFrame(rows, columns=["pid", "arm", "log2", "overlap_bp"])
    raw["w_log2"] = raw["log2"] * raw["overlap_bp"]
    per_pa = (raw.groupby(["pid", "arm"])
                  .agg(w_sum=("w_log2", "sum"), w_total=("overlap_bp", "sum"))
                  .reset_index())
    per_pa["mean_log2"] = per_pa["w_sum"] / per_pa["w_total"]
    log2_mat = (per_pa.pivot(index="pid", columns="arm", values="mean_log2")
                       .reindex(patients).fillna(0.0))

    out = {}
    for j, arm_label in enumerate(features):
        if not is_arm[j]:
            continue
        if arm_label not in log2_mat.columns:
            out[arm_label] = {"rho": 0.0, "P": 1.0, "sign": "?"}
            continue
        a_col = A[:, j]
        l_col = log2_mat[arm_label].values
        if (np.abs(a_col) < 1e-10).all():
            out[arm_label] = {"rho": 0.0, "P": 1.0, "sign": "?"}
            continue
        rho, p = spearmanr(a_col, l_col)
        sign = "AMP" if rho > 0 else "LOSS"
        out[arm_label] = {"rho": float(rho), "P": float(p), "sign": sign}
    return out


def label_signature_feature(
    feature: str, v: float, is_arm: bool, arm_dir: dict[str, dict] | None,
) -> str:
    """Build a biology-readable label for a single signature feature.

    For SNVs:  "TP53(+0.745)"     - gene-name + signed loading
    For arms:  "17p_LOSS(+0.258)" - arm-direction + signed loading

    The arm direction is determined by sign(v) × sign(rho), where rho is the
    Spearman correlation between the arm's attribution and its mean log2 ratio.
    """
    if not is_arm:
        return f"{feature}({v:+.3f})"
    if arm_dir is None or feature not in arm_dir or arm_dir[feature]["sign"] == "?":
        return f"{feature}({v:+.3f})"
    rho = arm_dir[feature]["rho"]
    direction = "AMP" if (v * rho > 0) else "LOSS"
    return f"{feature}_{direction}({v:+.3f})"
    # legacy code below (kept inert)
    base = arm_dir[feature]["sign"]
    if v < 0:
        flipped = {"AMP": "LOSS", "LOSS": "AMP"}[base]
        return f"{feature}_{flipped}({v:+.3f})"
    return f"{feature}_{base}({v:+.3f})"
