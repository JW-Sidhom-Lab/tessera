"""DepMap CRC cell-line validation of the MSK-CHORD CRC predictive biomarker.

Computes the FOLFOX-over-FOLFIRI z-scored sensitivity preference per CRC
cell line and tests its Spearman correlation with the MSK-CHORD-trained
tau_hat score (one-sided, predicted positive direction). Backs Fig. 6 n.

For each CRC cell line with both oxaliplatin AUC and SN-38 AUC measured:

    FOLFOX_sens_z   = -z(oxaliplatin AUC)
    FOLFIRI_sens_z  = -z(SN-38 AUC)
    preference      = FOLFOX_sens_z - FOLFIRI_sens_z   (>0: prefers FOLFOX)

Z-scoring uses the cohort mean and std of each drug independently, over
the cohort cell lines that have that drug measured.

Inputs:
    results/tau/depmap_tau.tsv   (per-cell-line tau_hat from apply_tau.py;
                                   joined with metadata + drug AUCs)

Outputs (results/analysis/):
    preference_per_cell_line.tsv   per cell line: tau_hat, FOLFOX_sens_z,
                                    FOLFIRI_sens_z, preference, lineage,
                                    primary_disease
    preference_test.tsv            single row: n, rho, P_two_sided,
                                    P_one_sided
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
TAU_TSV = ROOT / "results" / "tau" / "depmap_tau.tsv"
OUT     = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

CRC_PRIMARY_DISEASE = "Colorectal Adenocarcinoma"


def preference_test(df: pd.DataFrame):
    """Compute per-cell-line FOLFOX-over-FOLFIRI preference + Spearman test.

    Returns (per_cell_line_df, summary_df). The Spearman test is one-sided
    in the predicted positive direction (high tau_hat -> high preference).
    """
    crc = df[df["OncotreePrimaryDisease"] == CRC_PRIMARY_DISEASE].copy()

    ox = crc.dropna(subset=["oxaliplatin_AUC"])
    sn = crc.dropna(subset=["SN-38_AUC"])
    ox_mean, ox_std = float(ox["oxaliplatin_AUC"].mean()), float(ox["oxaliplatin_AUC"].std())
    sn_mean, sn_std = float(sn["SN-38_AUC"].mean()),       float(sn["SN-38_AUC"].std())

    both = crc.dropna(subset=["oxaliplatin_AUC", "SN-38_AUC", "tau_hat"]).copy()
    both["FOLFOX_sens_z"]  = -((both["oxaliplatin_AUC"] - ox_mean) / ox_std)
    both["FOLFIRI_sens_z"] = -((both["SN-38_AUC"]       - sn_mean) / sn_std)
    both["preference"]     = both["FOLFOX_sens_z"] - both["FOLFIRI_sens_z"]

    n = len(both)
    if n < 5:
        rho = np.nan; p_two = np.nan; p_dir = np.nan
    else:
        res = stats.spearmanr(both["tau_hat"], both["preference"])
        rho = float(res.statistic)
        p_two = float(res.pvalue)
        p_dir = p_two / 2 if rho > 0 else 1 - p_two / 2

    per_line = both[[
        "ModelID", "tau_hat",
        "FOLFOX_sens_z", "FOLFIRI_sens_z", "preference",
    ]].copy()
    per_line["lineage"]         = both.get("OncotreeLineage", np.nan).values
    per_line["primary_disease"] = both.get("OncotreePrimaryDisease", np.nan).values

    summary = pd.DataFrame([{
        "n":           int(n),
        "rho":         rho,
        "P_two_sided": p_two,
        "alternative": "greater",
        "P_one_sided": p_dir,
    }])

    return per_line.reset_index(drop=True), summary


def main():
    print("=" * 72)
    print("DepMap CRC cell-line validation: tau_hat vs FOLFOX-over-FOLFIRI preference")
    print("=" * 72)

    if not TAU_TSV.exists():
        raise FileNotFoundError(f"Missing {TAU_TSV}. Run apply_tau.py first.")
    df = pd.read_csv(TAU_TSV, sep="\t")
    print(f"Loaded {len(df):,} cell lines from {TAU_TSV.name}")

    per_line, summary = preference_test(df)

    p1 = OUT / "preference_per_cell_line.tsv"
    p2 = OUT / "preference_test.tsv"
    per_line.to_csv(p1, sep="\t", index=False)
    summary.to_csv(p2, sep="\t", index=False)
    print(f"  -> {p1}  ({len(per_line)} rows)")
    print(f"  -> {p2}  ({len(summary)} rows)")

    row = summary.iloc[0]
    print(f"\nCRC: n={int(row['n'])}, rho={row['rho']:+.4f}, "
          f"P_one_sided={row['P_one_sided']:.4f}")


if __name__ == "__main__":
    main()
