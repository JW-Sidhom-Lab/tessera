"""Continuous joint Cox of TESSERA risk score vs published comparator.

For BRCA and PRAD, fits

    hazard ~ z(comparator) + z(log(Risk_Score))

where ``comparator`` is the continuous DLRS-reconstructed OncotypeDX
Recurrence Score (BRCA) or the curatedPCaData Decipher score (PRAD),
and ``Risk_Score`` is the TESSERA polynomial-Cox partial hazard saved
per-sample by ``cohort_analysis.py``. Both predictors are standardised
to unit variance within the analytic cohort. Reports HR per +1 SD with
95% Wald CI, two-sided P, joint C-index, and Spearman rank correlation
between the two scores.

These are the numbers reported in the manuscript Figure 5 c-j (BRCA)
and Figure 5 k-r (PRAD) joint-Cox analyses.

Inputs
------
``analysis_results_brca_infonce_per_sample_loss/data/fig5_cache.csv``
``analysis_results_prad_infonce_per_sample_loss/data/fig5_cache.csv``
    Per-sample cache produced by ``cohort_analysis.py`` for each cohort
    (UMAP coords, Risk_Score, survival, Subtype tertile).

``../../data/prognostic/brca/brca_clinical_metadata.csv``
``../../data/prognostic/prad/prad_clinical_metadata.csv``
    Continuous comparator scores joined per ``Patient_ID``.

Output
------
``analysis_results_brca_infonce_per_sample_loss/joint_cox_continuous_brca_prad.txt``
    A small text artefact summarising both cohorts.

Usage
-----
    ./run_cohort.sh brca && ./run_cohort.sh prad   # produces fig5_cache for each
    python joint_cox.py                            # then this
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import spearmanr

HERE = Path(__file__).parent.resolve()

# Per-cohort wiring: (cache_results_dir, comparator_score_col, time_col, event_col, comparator_label)
COHORTS = [
    ("brca",
     HERE / "analysis_results_brca_infonce_per_sample_loss",
     HERE / ".." / ".." / "data" / "prognostic" / "brca" / "brca_clinical_metadata.csv",
     "oncotype_rs_dlrs", "DSS.time", "DSS",
     "DSS, 5y", "Oncotype"),
    ("prad",
     HERE / "analysis_results_prad_infonce_per_sample_loss",
     HERE / ".." / ".." / "data" / "prognostic" / "prad" / "prad_clinical_metadata.csv",
     "decipher_curatedpcadata", "PFI.time.1", "PFI.1",
     "PFI, 5y", "Decipher"),
]
TIME_LIMIT_DAYS = 1825   # 5-year administrative censoring (matches cohort configs)
OUTPUT_TXT = Path(os.environ.get(
    "OUTPUT_TXT",
    HERE / "analysis_results_brca_infonce_per_sample_loss"
         / "joint_cox_continuous_brca_prad.txt"))


def _fit(cache_dir: Path, clinical_csv: Path, score_col: str,
         time_col: str, event_col: str):
    cache = pd.read_csv(cache_dir / "data" / "fig5_cache.csv")
    meta = pd.read_csv(clinical_csv)[["Patient_ID", score_col]]
    df = cache.merge(meta, on="Patient_ID", how="inner") \
              .dropna(subset=[score_col, "Risk_Score", time_col, event_col])
    df = df[df[time_col] > 0].copy()
    df["T"] = df[time_col].clip(upper=TIME_LIMIT_DAYS).astype(float)
    df["E"] = df[event_col].astype(int)
    df.loc[df[time_col] > TIME_LIMIT_DAYS, "E"] = 0
    df["comparator_z"] = ((df[score_col] - df[score_col].mean())
                          / df[score_col].std())
    df["foundation_z"] = ((np.log(df["Risk_Score"]) - np.log(df["Risk_Score"]).mean())
                          / np.log(df["Risk_Score"]).std())
    cph = CoxPHFitter().fit(df[["T", "E", "comparator_z", "foundation_z"]],
                            "T", "E")
    rho, _ = spearmanr(df[score_col], df["Risk_Score"])
    return df, cph, rho


def _format_block(cohort, endpoint_label, comparator_label,
                  df, cph, rho) -> str:
    lines = [f"{cohort.upper()} ({endpoint_label}) joint Cox  "
             f"n={len(df)} events={int(df['E'].sum())}  "
             f"C={cph.concordance_index_:.4f}"]
    for col, label in [("comparator_z", f"{comparator_label.lower()}_z"),
                       ("foundation_z", "foundation_z")]:
        s = cph.summary.loc[col]
        lines.append(
            f"  {label}: HR={float(s['exp(coef)']):.3f} "
            f"({float(s['exp(coef) lower 95%']):.3f}-"
            f"{float(s['exp(coef) upper 95%']):.3f}), "
            f"P={float(s['p']):.4g}")
    lines.append(f"  Spearman rho({comparator_label}, Foundation) = {rho:+.3f}")
    return "\n".join(lines)


def main():
    blocks = []
    for (cohort, cache_dir, clinical_csv, score, t_col, e_col,
         endpoint_label, comparator_label) in COHORTS:
        if not (cache_dir / "data" / "fig5_cache.csv").exists():
            print(f"[skip] {cohort}: no cache at {cache_dir}/data/fig5_cache.csv "
                  f"(run ./run_cohort.sh first)")
            continue
        df, cph, rho = _fit(cache_dir, clinical_csv, score, t_col, e_col)
        block = _format_block(cohort, endpoint_label, comparator_label,
                              df, cph, rho)
        print(block); print()
        blocks.append(block)

    if blocks:
        OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_TXT.write_text("\n\n".join(blocks) + "\n")
        print(f"wrote {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
