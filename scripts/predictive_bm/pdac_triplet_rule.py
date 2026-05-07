"""PDAC Sig 8 (1-indexed) triplet-rule validation.

Sig 8 of the PDAC PMD decomposition at K=10 is dominated by three biology
signals (positive directional loadings on the FFX axis): 10q amplification,
TP53 mutation, and 17p intact (negative loading on 17p_LOSS). Rather than
report 10q amplification at the strict log2 > +0.2 threshold (only 19/771
patients), the manifest rule swaps in 20q amplification — the next-largest
positive arm loading on Sig 8 (+0.16 directional) and a much more prevalent
event in PDAC (~21%). This trades a small loading-magnitude reduction for
a much larger and more clinically usable subgroup.

The triplet rule is therefore:

    TP53 coding mutation
    AND 17p intact            (length-weighted log2 segment-mean >= -0.2)
    AND 20q amplified         (length-weighted log2 segment-mean >  +0.2)

This script computes per-stratum PFS arm hazard ratios for the full triplet
plus its component subsets, used as the data backing the PDAC supplementary
figure (panels c, d, e).

Outputs:
  attribution_analysis/pdac_signatures/triplet_rule/
    triplet_per_group_metrics.tsv  — progressive subgroup forest data
    panelD_triplet_rule_KM.tsv     — per-patient table for triplet KM
    panelE_triplet_rest_KM.tsv     — per-patient table for "none of three" KM
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.dr import per_group_arm_hr
from core.arm_mapping import build_arm_coords


OUT = ROOT / "attribution_analysis" / "pdac_signatures"
VAL = OUT / "triplet_rule"
VAL.mkdir(parents=True, exist_ok=True)


def per_patient_arm_log2(cna_meta: pd.DataFrame, arm_label: str,
                          patients: list) -> pd.Series:
    """Length-weighted mean Segment_Mean per patient for a chromosomal arm."""
    arms = build_arm_coords()
    arm_row = arms[arms["arm"] == arm_label].iloc[0]
    arm_chrom = str(arm_row["chr"])
    arm_start = int(arm_row["start"])
    arm_end = int(arm_row["end"])
    rows = []
    for _, seg in cna_meta[cna_meta["Chromosome"].astype(str) == arm_chrom].iterrows():
        ov = max(0, min(int(seg["End"]), arm_end) - max(int(seg["Start"]), arm_start))
        if ov > 0:
            rows.append((seg["pid"], seg["Segment_Mean"], ov))
    raw = pd.DataFrame(rows, columns=["pid", "log2", "ov"])
    raw["w"] = raw["log2"] * raw["ov"]
    per = (raw.groupby("pid")
              .agg(s=("w", "sum"), w=("ov", "sum"))
              .reset_index())
    per["log2"] = per["s"] / per["w"]
    return per.set_index("pid")["log2"].reindex(patients).fillna(0.0)


def safe_arm_hr(df, mask, *, time_col, event_col):
    """Cox PFS ~ arm with NaN-safe handling for degenerate strata."""
    sub = df[mask]
    nan = {"n": int(len(sub)), "HR": float("nan"), "HR_lo": float("nan"),
            "HR_hi": float("nan"), "P": float("nan")}
    if len(sub) < 10:
        return nan
    arm_counts = sub["arm"].value_counts()
    if len(arm_counts) < 2 or (arm_counts < 5).any():
        return nan
    if sub[event_col].sum() < 3:
        return nan
    try:
        return per_group_arm_hr(df, mask, time_col=time_col, event_col=event_col)
    except Exception:
        return nan


def main():
    state = pickle.load(open(OUT / "attribution_matrix.pkl", "rb"))
    df = state["df"]
    patients = state["patients"]

    bundle = pickle.load(open(ROOT / "msk_chord_latent_features_raw.pkl", "rb"))
    snv_meta = bundle["data_snv"].copy()
    snv_meta["pid"] = snv_meta["Tumor_Sample_Barcode"].str[:9]
    cna_meta = bundle["data_cna"].copy()
    cna_meta["pid"] = cna_meta["Tumor_Sample_Barcode"].str[:9]

    # Per-patient indicators
    tp53 = pd.Series(0, index=patients, dtype=int)
    for pid in snv_meta[snv_meta["Hugo_Symbol"] == "TP53"]["pid"].unique():
        if pid in tp53.index:
            tp53[pid] = 1
    intact_17p = (per_patient_arm_log2(cna_meta, "17p", patients) >= -0.2).astype(int)
    amp_20q    = (per_patient_arm_log2(cna_meta, "20q", patients) >   0.2).astype(int)

    # Progressive subgroups for the panel c forest, ordered triplet → subsets.
    groups = {
        "TP53+ / 17p intact / 20q+":               (tp53.values == 1) & (intact_17p.values == 1) & (amp_20q.values == 1),
        "TP53+ / 20q+ (any 17p)":                  (tp53.values == 1) & (amp_20q.values == 1),
        "17p intact / 20q+ (no TP53)":             (tp53.values == 0) & (intact_17p.values == 1) & (amp_20q.values == 1),
        "TP53+ alone (no 17p, no 20q gate)":       (tp53.values == 1) & (intact_17p.values == 0) & (amp_20q.values == 0),
        "TP53+ / 17p intact (no 20q)":             (tp53.values == 1) & (intact_17p.values == 1) & (amp_20q.values == 0),
        "None of the three (rule-disfavoured)":    (tp53.values == 0) & ((intact_17p.values == 0) | (amp_20q.values == 0)),
    }

    print("=" * 78)
    print("PDAC Sig 8 triplet rule — per-stratum PFS arm HR (FFX vs GA)")
    print("=" * 78)
    rows = []
    for label, mask in groups.items():
        hr_pfs = safe_arm_hr(df, mask, time_col="pfs_t", event_col="pfs_e")
        hr_os  = safe_arm_hr(df, mask, time_col="os_t",  event_col="os_e")
        rows.append({
            "group": label,
            "n": hr_pfs["n"],
            "PFS_HR": hr_pfs["HR"], "PFS_HR_lo": hr_pfs["HR_lo"],
            "PFS_HR_hi": hr_pfs["HR_hi"], "PFS_P": hr_pfs["P"],
            "OS_HR": hr_os["HR"], "OS_P": hr_os["P"],
        })
        favor = ("FFX" if hr_pfs["HR"] < 1
                  else "GA" if hr_pfs["HR"] > 1
                  else "n/a") if np.isfinite(hr_pfs["HR"]) else "n/a"
        print(f"\n{label}  n={hr_pfs['n']:4d}")
        print(f"  PFS  HR(FFX vs GA) = {hr_pfs['HR']:.3f} "
              f"[{hr_pfs['HR_lo']:.3f}, {hr_pfs['HR_hi']:.3f}]  "
              f"P={hr_pfs['P']:.3g}  -> {favor}")

    pd.DataFrame(rows).to_csv(VAL / "triplet_per_group_metrics.tsv",
                                  sep="\t", index=False)
    print(f"\n→ {VAL / 'triplet_per_group_metrics.tsv'}")

    # Per-patient TSVs for the supplementary-figure KM panels
    rule_match = (tp53.values == 1) & (intact_17p.values == 1) & (amp_20q.values == 1)
    rule_disfavoured = (tp53.values == 0) & ((intact_17p.values == 0) | (amp_20q.values == 0))

    match_pids = set(np.array(patients)[rule_match])
    rest_pids  = set(np.array(patients)[rule_disfavoured])
    keep_cols  = ["pid", "arm", "pfs_t", "pfs_e"]
    df[df["pid"].isin(match_pids)][keep_cols].to_csv(
        VAL / "panelD_triplet_rule_KM.tsv", sep="\t", index=False)
    df[df["pid"].isin(rest_pids)][keep_cols].to_csv(
        VAL / "panelE_triplet_rest_KM.tsv", sep="\t", index=False)
    print(f"→ {VAL / 'panelD_triplet_rule_KM.tsv'}")
    print(f"→ {VAL / 'panelE_triplet_rest_KM.tsv'}")


if __name__ == "__main__":
    main()
