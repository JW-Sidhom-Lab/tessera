"""PDAC FOLFIRINOX vs Gem/Abraxane predictive biomarker.

Fits the same doubly-robust learner pipeline as crc_discovery.py on
MSK-CHORD 1L stage IV PDAC. Writes the predictions CSV consumed by
build_figure6_panels.py to produce Fig 6 f-i and Sup Fig 10 c-d.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.features import build_patient_features
from core.cohorts import build_pdac_met
from core.dr import (fit_dr_learner, interaction_test,
                       threshold_indifference, per_group_arm_hr)

DATA = ROOT / ".." / ".." / "data" / "msk_chord_2024"
GT = DATA / "GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv"
RAW_PKL = ROOT / "msk_chord_latent_features_raw.pkl"
CACHE = ROOT / "cache" / "patient_features.pkl"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# PDAC PCA(0.99) coord-descent optimized config (validated final)
PCA_VAR = 0.99
K_MU1, SP_MU1 = 2, 0.5
K_MU0, SP_MU0 = 3, 0.85
K_E,   SP_E   = 2, 0.0
K_TAU, SP_TAU = 5, 0.85
N_OUTER = 10
N_INNER = 5
SEED    = 42
HORIZON = 36.0


def banner(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def main():
    t0 = time.time()
    banner("Step 1 - load patient features (cache reuse)")
    feat = build_patient_features(RAW_PKL, cache=CACHE)
    print(f"  feature matrix shape: {feat.shape}")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    banner("Step 2 - build PDAC stage IV first-line cohort")
    df, X = build_pdac_met(GT, feat, horizon_months=HORIZON)
    arm_counts = df["arm"].value_counts().to_dict()
    print(f"  cohort n={len(df)}")
    print(f"  arm counts: FOLFIRINOX(arm=1)={arm_counts.get(1,0)}  "
          f"GemAbraxane(arm=0)={arm_counts.get(0,0)}")
    print(f"  PFS events: {int(df['pfs_e'].sum())}/{len(df)}")
    print(f"  OS  events: {int(df['os_e'].sum())}/{len(df)}")
    print(f"  X shape: {X.shape}")

    banner("Step 3 - fit DR-learner (nested 10x5 CV)")
    print(f"  PCA={PCA_VAR}  K_mu1={K_MU1} sp={SP_MU1}  K_mu0={K_MU0} sp={SP_MU0}")
    print(f"  K_e={K_E} sp={SP_E}  K_tau={K_TAU} sp={SP_TAU}  seed={SEED}")
    t1 = time.time()
    res = fit_dr_learner(
        df, X,
        pca_var=PCA_VAR,
        K_mu1=K_MU1, K_mu0=K_MU0, K_e=K_E, K_tau=K_TAU,
        sp_mu1=SP_MU1, sp_mu0=SP_MU0, sp_e=SP_E, sp_tau=SP_TAU,
        n_outer=N_OUTER, n_inner=N_INNER, seed=SEED,
        horizon_months=HORIZON,
    )
    print(f"  DR-learner done in {time.time()-t1:.1f}s")
    print(f"  PCA components used: {res['n_pca_components']}")

    banner("Step 4 - sign-align τ̂")
    raw = res["tau_oof"]
    itx_raw = interaction_test(df, raw, time_col="pfs_t", event_col="pfs_e")
    sign = -1.0 if itx_raw["HR"] > 1.0 else +1.0
    tau = sign * raw
    df["tau"] = tau
    print(f"  raw interaction HR={itx_raw['HR']:.3f}; sign-flip={sign:+.0f}")

    banner("Step 5 - interaction Cox + per-stratum HRs")
    itx_pfs = interaction_test(df, tau, time_col="pfs_t", event_col="pfs_e")
    itx_os  = interaction_test(df, tau, time_col="os_t",  event_col="os_e")
    print(f"  PFS interaction: HR = {itx_pfs['HR']:.3f} "
          f"[{itx_pfs['HR_lo']:.3f}, {itx_pfs['HR_hi']:.3f}]   P = {itx_pfs['P']:.3e}")
    print(f"  OS  interaction: HR = {itx_os['HR']:.3f} "
          f"[{itx_os['HR_lo']:.3f}, {itx_os['HR_hi']:.3f}]   P = {itx_os['P']:.3e}")

    tau0 = threshold_indifference(itx_pfs)
    above = (tau > tau0)
    below = ~above
    print(f"\n  Indifference threshold τ̂_0 (from PFS interaction Cox) = {tau0:+.3f}")
    print(f"  ABOVE n={int(above.sum())} ({100*above.mean():.1f}%)   "
          f"BELOW n={int(below.sum())} ({100*below.mean():.1f}%)")

    a_pfs = per_group_arm_hr(df, above, time_col="pfs_t", event_col="pfs_e")
    b_pfs = per_group_arm_hr(df, below, time_col="pfs_t", event_col="pfs_e")
    a_os  = per_group_arm_hr(df, above, time_col="os_t",  event_col="os_e")
    b_os  = per_group_arm_hr(df, below, time_col="os_t",  event_col="os_e")
    print()
    print(f"  ABOVE PFS  n={a_pfs['n']:4d}   HR(FFX vs GA) = {a_pfs['HR']:.3f} "
          f"[{a_pfs['HR_lo']:.3f}, {a_pfs['HR_hi']:.3f}]   P = {a_pfs['P']:.3e}")
    print(f"  BELOW PFS  n={b_pfs['n']:4d}   HR(FFX vs GA) = {b_pfs['HR']:.3f} "
          f"[{b_pfs['HR_lo']:.3f}, {b_pfs['HR_hi']:.3f}]   P = {b_pfs['P']:.3e}")
    print(f"  ABOVE OS   n={a_os['n']:4d}   HR(FFX vs GA) = {a_os['HR']:.3f} "
          f"[{a_os['HR_lo']:.3f}, {a_os['HR_hi']:.3f}]   P = {a_os['P']:.3e}")
    print(f"  BELOW OS   n={b_os['n']:4d}   HR(FFX vs GA) = {b_os['HR']:.3f} "
          f"[{b_os['HR_lo']:.3f}, {b_os['HR_hi']:.3f}]   P = {b_os['P']:.3e}")

    out_csv = RESULTS / "pdac_predictions.csv"
    df_save = df[["pid", "PATIENT_ID", "REGIMEN", "arm",
                    "pfs_t", "pfs_e", "os_t", "os_e", "tau"]].copy()
    df_save["above_threshold"] = above.astype(int)
    df_save.to_csv(out_csv, index=False)
    print(f"\n✓ saved per-patient predictions to {out_csv}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
