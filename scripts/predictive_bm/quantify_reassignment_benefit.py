"""Counterfactual treatment-reassignment analysis for the CRC predictive
biomarker, projected onto the SEER-Medicare 1L stage IV colon cancer
treatment distribution (Neugut et al. 2019, PMID 30878317).

Question: in real-world practice (79.3% FOLFOX, 20.7% FOLFIRI per
SEER-Medicare), what fraction of patients would be re-assigned to a
different regimen by the TESSERA biomarker, and what is the per-patient
median PFS and OS benefit for re-assigned patients?

Method:
  1. Compute biomarker prevalence p_above = P(τ̂ > τ̂_0) in our cohort.
  2. Apply SEER-Medicare base rate (79.3% FOLFOX, 20.7% FOLFIRI),
     assuming biomarker classification is independent of current
     treatment assignment (consistent with Neugut 2019's null OS
     finding — selection by toxicity not biology).
  3. Compute reassignment percentages:
        FOLFOX→FOLFIRI = 79.3% × (1 − p_above)
        FOLFIRI→FOLFOX = 20.7% × p_above
  4. For each reassignment direction, compute the within-stratum
     cross-arm median Δ for PFS and OS — the counterfactual benefit
     a re-assigned patient would gain. The DR-learner τ̂_0 partition
     was specifically designed to remove treatment-assignment
     confounding via the doubly-robust pseudo-outcome, so within-
     stratum cross-arm comparisons are the right counterfactual proxy.

Outputs: methods/predictive_bm/results/reassignment_benefit.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# SEER-Medicare base rates (Neugut 2019, PMID 30878317; n=3,785, age >=65)
SEER_FOLFOX = 0.793
SEER_FOLFIRI = 0.207


def per_arm_median(df: pd.DataFrame, time_col: str, event_col: str) -> dict:
    """Return per-arm median (and lower/upper 95% CI bounds, NaN if not reached)."""
    out = {}
    for arm_v in (0, 1):
        sub = df[df["arm"] == arm_v]
        kmf = KaplanMeierFitter().fit(
            sub[time_col].values, event_observed=sub[event_col].values)
        med = float(kmf.median_survival_time_)
        try:
            ci_df = kmf.confidence_interval_survival_function_
            below_half = ci_df[ci_df.iloc[:, 0] <= 0.5].index
            above_half = ci_df[ci_df.iloc[:, 1] <= 0.5].index
            ci_lo = float(below_half[0]) if len(below_half) else float("nan")
            ci_hi = float(above_half[0]) if len(above_half) else float("nan")
        except Exception:
            ci_lo, ci_hi = float("nan"), float("nan")
        out[arm_v] = {
            "n": int(len(sub)),
            "n_events": int(sub[event_col].sum()),
            "median": med,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    return out


def banner(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)


def main():
    df = pd.read_csv(ROOT / "results" / "crc_predictions.csv")
    print(f"loaded CRC predictions: {len(df)} patients")

    # Step 1 — biomarker prevalence
    n_total = len(df)
    n_above = int((df["above_threshold"] == 1).sum())
    n_below = int((df["above_threshold"] == 0).sum())
    p_above = n_above / n_total
    p_below = n_below / n_total
    banner(f"Biomarker prevalence in CRC cohort (n={n_total})")
    print(f"  predicted-FOLFOX-favoured (above τ̂_0):  n={n_above}  ({p_above*100:.1f}%)")
    print(f"  predicted-FOLFIRI-favoured (below τ̂_0): n={n_below}  ({p_below*100:.1f}%)")

    # Step 2 — reassignment percentages projected to SEER-Medicare
    pct_FtoFI = SEER_FOLFOX  * p_below      # FOLFOX recipient, FOLFIRI-favoured by biomarker
    pct_FItoF = SEER_FOLFIRI * p_above      # FOLFIRI recipient, FOLFOX-favoured by biomarker
    pct_total = pct_FtoFI + pct_FItoF

    banner("Reassignment percentages (assuming SEER-Medicare base rate)")
    print(f"  SEER-Medicare base: 79.3% FOLFOX, 20.7% FOLFIRI (Neugut 2019)")
    print(f"  Reassign FOLFOX → FOLFIRI:  {pct_FtoFI*100:.1f}%   (= 79.3% × {p_below*100:.1f}%)")
    print(f"  Reassign FOLFIRI → FOLFOX:  {pct_FItoF*100:.1f}%   (= 20.7% × {p_above*100:.1f}%)")
    print(f"  Total reassigned:           {pct_total*100:.1f}%")

    # Step 3 — within-stratum per-arm medians + Δ
    # FOLFOX → FOLFIRI re-assigned patients land in the predicted-FOLFIRI-
    # favoured stratum (below τ̂_0); their counterfactual benefit is the
    # median Δ between FOLFIRI and FOLFOX in that stratum.
    below = df[df["above_threshold"] == 0]
    above = df[df["above_threshold"] == 1]

    rows = []
    for stratum_name, sub in [("below", below), ("above", above)]:
        for endpoint, time_col, event_col in [("PFS", "pfs_t", "pfs_e"),
                                                ("OS",  "os_t",  "os_e")]:
            meds = per_arm_median(sub, time_col, event_col)
            row = {
                "stratum": stratum_name,
                "endpoint": endpoint,
                "n_total": int(len(sub)),
                "n_arm1_FOLFOX":  meds[1]["n"],
                "n_arm0_FOLFIRI": meds[0]["n"],
                "events_arm1":    meds[1]["n_events"],
                "events_arm0":    meds[0]["n_events"],
                "median_FOLFOX":  meds[1]["median"],
                "median_FOLFOX_ci_lo": meds[1]["ci_lo"],
                "median_FOLFOX_ci_hi": meds[1]["ci_hi"],
                "median_FOLFIRI": meds[0]["median"],
                "median_FOLFIRI_ci_lo": meds[0]["ci_lo"],
                "median_FOLFIRI_ci_hi": meds[0]["ci_hi"],
            }
            # Δ-median = median(arm-favoured-by-biomarker) - median(other arm)
            if stratum_name == "above":
                # Above τ̂_0 = FOLFOX-favoured. Reassignment FOLFIRI→FOLFOX.
                # Benefit = median(FOLFOX) - median(FOLFIRI) for these patients.
                delta = meds[1]["median"] - meds[0]["median"]
                row["delta_median_recommended_minus_other"] = delta
                row["reassignment_direction"] = "FOLFIRI→FOLFOX"
            else:
                # Below τ̂_0 = FOLFIRI-favoured. Reassignment FOLFOX→FOLFIRI.
                # Benefit = median(FOLFIRI) - median(FOLFOX) for these patients.
                delta = meds[0]["median"] - meds[1]["median"]
                row["delta_median_recommended_minus_other"] = delta
                row["reassignment_direction"] = "FOLFOX→FOLFIRI"
            rows.append(row)

    summary = pd.DataFrame(rows)

    banner("Within-stratum per-arm medians and Δ (counterfactual benefit)")
    for _, r in summary.iterrows():
        print(f"\n  Stratum: {r['stratum']} τ̂_0  | Endpoint: {r['endpoint']}  "
              f"(reassignment {r['reassignment_direction']})")
        print(f"    n = {r['n_total']}  "
              f"(FOLFOX n={r['n_arm1_FOLFOX']}, FOLFIRI n={r['n_arm0_FOLFIRI']})")
        med_fox = r['median_FOLFOX']; med_fri = r['median_FOLFIRI']
        print(f"    median FOLFOX  = {med_fox:.1f} mo  [{r['median_FOLFOX_ci_lo']:.1f}, {r['median_FOLFOX_ci_hi']:.1f}]")
        print(f"    median FOLFIRI = {med_fri:.1f} mo  [{r['median_FOLFIRI_ci_lo']:.1f}, {r['median_FOLFIRI_ci_hi']:.1f}]")
        print(f"    Δ median (recommended − other) = {r['delta_median_recommended_minus_other']:+.1f} mo")

    # Save
    summary.to_csv(OUT / "reassignment_benefit.tsv", sep="\t", index=False)
    print(f"\n→ {OUT / 'reassignment_benefit.tsv'}")

    # Step 4 — population-projected summary
    banner("Population-scale summary (projected to SEER-Medicare distribution)")
    print(f"  {pct_total*100:.1f}% of patients re-assigned overall:")
    print(f"    {pct_FtoFI*100:.1f}% FOLFOX → FOLFIRI:")
    pfs_below = summary[(summary['stratum']=='below') & (summary['endpoint']=='PFS')].iloc[0]
    os_below  = summary[(summary['stratum']=='below') & (summary['endpoint']=='OS')].iloc[0]
    print(f"      ΔPFS = {pfs_below['delta_median_recommended_minus_other']:+.1f} mo")
    print(f"      ΔOS  = {os_below['delta_median_recommended_minus_other']:+.1f} mo")
    print(f"    {pct_FItoF*100:.1f}% FOLFIRI → FOLFOX:")
    pfs_above = summary[(summary['stratum']=='above') & (summary['endpoint']=='PFS')].iloc[0]
    os_above  = summary[(summary['stratum']=='above') & (summary['endpoint']=='OS')].iloc[0]
    print(f"      ΔPFS = {pfs_above['delta_median_recommended_minus_other']:+.1f} mo")
    print(f"      ΔOS  = {os_above['delta_median_recommended_minus_other']:+.1f} mo")


if __name__ == "__main__":
    main()
