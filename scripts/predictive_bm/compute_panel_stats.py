"""Aggregate per-patient predictions into per-stratum survival statistics.

Reads the prediction CSVs written by ``crc_discovery.py`` and
``pdac_discovery.py`` (one row per patient with ``tau``, ``arm``,
PFS, OS) and computes, for each cohort:

  - the indifference threshold tau_0 (from the interaction Cox)
  - per-stratum cross-arm Cox HRs (HR + CI + P) and logrank P-values
  - per-stratum Kaplan-Meier curves
  - a sliding-window HR(tau) curve along the tau axis
  - a smooth analytic HR(tau) curve from the interaction Cox
  - the pooled-cohort split as a prognostic-null check

The same aggregation is run for the three feature-slice ablations
(Local / +Global / +InfoNCE) feeding Sup Fig 11.

Outputs, per cohort, under ``results/figures/<cohort>/figure6/``:

  panels.npz     KM curves + HR/CI/P scalars + HR(tau) curves (full feature run)
  ablation.npz   same statistics for the three ablation slices
  meta.json      tau_0, sample sizes, axis limits

No matplotlib; the bundles are the data backing the published Fig 6
and Sup Fig 11 panels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dr import interaction_test, threshold_indifference

RESULTS = ROOT / "results"
ABL_DIR = RESULTS / "ablation"

COHORTS = {
    "crc": dict(
        pred_csv=RESULTS / "crc_predictions.csv",
        ablation=dict(
            local       = ABL_DIR / "crc_predictions_local.csv",
            local_global= ABL_DIR / "crc_predictions_local_global.csv",
            full        = ABL_DIR / "crc_predictions_full.csv",
        ),
        arm1_label="FOLFOX", arm0_label="FOLFIRI",
        pfs_xlim=24.0, os_xlim=60.0,
    ),
    "pdac": dict(
        pred_csv=RESULTS / "pdac_predictions.csv",
        ablation=dict(
            local       = ABL_DIR / "pdac_predictions_local.csv",
            local_global= ABL_DIR / "pdac_predictions_local_global.csv",
            full        = ABL_DIR / "pdac_predictions_full.csv",
        ),
        arm1_label="FOLFIRINOX", arm0_label="Gem-Abraxane",
        pfs_xlim=24.0, os_xlim=60.0,
    ),
}


def _arm_hr(sub: pd.DataFrame, time_col: str, event_col: str) -> dict:
    """Cox HR(arm=1 vs arm=0) within a subset."""
    if sub["arm"].nunique() < 2 or len(sub) < 5:
        return dict(HR=np.nan, HR_lo=np.nan, HR_hi=np.nan, P=np.nan)
    cph = CoxPHFitter().fit(sub[[time_col, event_col, "arm"]],
                            duration_col=time_col, event_col=event_col)
    s = cph.summary.loc["arm"]
    return dict(HR=float(s["exp(coef)"]),
                HR_lo=float(s["exp(coef) lower 95%"]),
                HR_hi=float(s["exp(coef) upper 95%"]),
                P=float(s["p"]))


def _km_curve(times: pd.Series, events: pd.Series) -> dict:
    """Step KM curve at event times + CI ribbon."""
    if len(times) < 1:
        return dict(t=np.array([]), s=np.array([]), s_lo=np.array([]), s_hi=np.array([]),
                    n=0, n_events=0)
    km = KaplanMeierFitter().fit(times.values, events.values)
    sf = km.survival_function_.iloc[:, 0]
    ci = km.confidence_interval_
    return dict(
        t=sf.index.values.astype(float),
        s=sf.values.astype(float),
        s_lo=ci.iloc[:, 0].values.astype(float),
        s_hi=ci.iloc[:, 1].values.astype(float),
        n=int(len(times)),
        n_events=int(events.sum()),
    )


def _sliding_window_hr(df: pd.DataFrame, time_col: str, event_col: str, *,
                        window_frac=0.30, n_points=50):
    """Per-percentile sliding-window HR curve along τ̂."""
    df = df.sort_values("tau").reset_index(drop=True)
    n = len(df)
    win = max(int(window_frac * n), 30)
    points = np.linspace(win // 2, n - win // 2 - 1, n_points).astype(int)
    HRs, los, his, taus = [], [], [], []
    for i in points:
        lo, hi = max(0, i - win // 2), min(n, i + win // 2)
        sub = df.iloc[lo:hi]
        h = _arm_hr(sub, time_col, event_col)
        HRs.append(h["HR"]); los.append(h["HR_lo"]); his.append(h["HR_hi"])
        taus.append(float(sub["tau"].median()))
    return (np.array(taus), np.array(HRs), np.array(los), np.array(his))


def _km_pair(sub: pd.DataFrame, time_col: str, event_col: str) -> dict:
    """KM for arm=1 + arm=0 + Cox HR + logrank P, packed for one panel."""
    arm1 = sub[sub["arm"] == 1]
    arm0 = sub[sub["arm"] == 0]
    h = _arm_hr(sub, time_col, event_col)
    lr_p = np.nan
    if len(arm1) >= 1 and len(arm0) >= 1:
        try:
            lr = logrank_test(arm1[time_col], arm0[time_col],
                              event_observed_A=arm1[event_col],
                              event_observed_B=arm0[event_col])
            lr_p = float(lr.p_value)
        except Exception:
            pass
    return dict(
        arm1=_km_curve(arm1[time_col], arm1[event_col]),
        arm0=_km_curve(arm0[time_col], arm0[event_col]),
        HR=h["HR"], HR_lo=h["HR_lo"], HR_hi=h["HR_hi"], P=h["P"],
        logrank_P=lr_p,
        n=int(len(sub)),
    )


def _flatten(prefix: str, panel: dict) -> dict:
    """Flatten nested KM-pair dict into npz-friendly arrays/scalars."""
    out = {}
    for arm in ("arm1", "arm0"):
        for k, v in panel[arm].items():
            out[f"{prefix}_{arm}_{k}"] = np.array(v) if isinstance(v, np.ndarray) else v
    for k in ("HR", "HR_lo", "HR_hi", "P", "logrank_P", "n"):
        out[f"{prefix}_{k}"] = panel[k]
    return out


def _annotate(df: pd.DataFrame, tau0: float) -> pd.DataFrame:
    df = df.copy()
    df["recommended"] = (df["tau"] > tau0).astype(int)
    df["tertile"] = (2 - pd.qcut(df["tau"].rank(method="first"),
                                   q=3, labels=False, duplicates="drop").astype(int))
    return df


def compute_main(df: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    """Compute all main-panel artifacts for one cohort."""
    itx_pfs = interaction_test(df, df["tau"].values, time_col="pfs_t", event_col="pfs_e")
    tau0 = threshold_indifference(itx_pfs)
    df = _annotate(df, tau0)

    above = df[df["recommended"] == 1]
    below = df[df["recommended"] == 0]

    # Row 1 col 1: τ̂ histogram per arm (just the raw arrays for plotting)
    arm1_tau = df[df["arm"] == 1]["tau"].values
    arm0_tau = df[df["arm"] == 0]["tau"].values
    bins = np.linspace(df["tau"].min(), df["tau"].max(), 60)
    hist_arm1, _ = np.histogram(arm1_tau, bins=bins)
    hist_arm0, _ = np.histogram(arm0_tau, bins=bins)

    # Row 1 col 1 (overlay): sliding-window HR curve on PFS (kept for context;
    # the panel itself uses the smooth analytic curve from the interaction Cox)
    hr_centers, hr_vals, hr_lo, hr_hi = _sliding_window_hr(
        df, "pfs_t", "pfs_e", window_frac=0.30, n_points=50)

    # Smooth analytic HR(τ̂) = exp(β_T + β_int · τ̂) from the PFS interaction Cox.
    # Also expose β_T, β_int and a per-tau std-error band derived from the Cox
    # variance-covariance matrix so the build script can draw a clean ribbon.
    from lifelines import CoxPHFitter
    work = pd.DataFrame({
        "T": df["pfs_t"].values,
        "E": df["pfs_e"].values,
        "tau": df["tau"].values,
        "arm": df["arm"].values.astype(int),
    })
    work["arm_tau"] = work["arm"] * work["tau"]
    cph = CoxPHFitter().fit(work[["T", "E", "tau", "arm", "arm_tau"]],
                              duration_col="T", event_col="E")
    beta_T   = float(cph.params_["arm"])
    beta_int = float(cph.params_["arm_tau"])
    cov = cph.variance_matrix_
    var_T   = float(cov.loc["arm", "arm"])
    var_int = float(cov.loc["arm_tau", "arm_tau"])
    cov_T_int = float(cov.loc["arm", "arm_tau"])

    tau_grid = np.linspace(arm0_tau.min(), arm1_tau.max(), 200)  # cohort range
    log_hr   = beta_T + beta_int * tau_grid
    se_log_hr = np.sqrt(var_T + (tau_grid ** 2) * var_int + 2 * tau_grid * cov_T_int)
    hr_smooth_HR = np.exp(log_hr)
    hr_smooth_lo = np.exp(log_hr - 1.96 * se_log_hr)
    hr_smooth_hi = np.exp(log_hr + 1.96 * se_log_hr)

    # Row 1 col 2: pooled cohort split by τ̂_0 (PFS only)
    pooled_pfs = _km_pair(df, "pfs_t", "pfs_e")  # for HR(arm) sanity
    # But the requested panel is splitting BY τ̂_0, so two curves: above vs below
    pooled_above = _km_curve(above["pfs_t"], above["pfs_e"])
    pooled_below = _km_curve(below["pfs_t"], below["pfs_e"])
    # logrank between above vs below in the pooled cohort (predictive label collapsed across arms)
    lr = logrank_test(above["pfs_t"], below["pfs_t"],
                      event_observed_A=above["pfs_e"],
                      event_observed_B=below["pfs_e"])
    pooled_lr_p = float(lr.p_value)
    # Cox HR(above vs below) on PFS, pooled (no arm adjustment) - should be ~1 if τ̂ is purely predictive
    cph = CoxPHFitter().fit(
        df.assign(above=df["recommended"])[["pfs_t","pfs_e","above"]],
        duration_col="pfs_t", event_col="pfs_e")
    pooled_HR    = float(np.exp(cph.params_["above"]))
    pooled_HR_lo = float(np.exp(cph.confidence_intervals_.loc["above", "95% lower-bound"]))
    pooled_HR_hi = float(np.exp(cph.confidence_intervals_.loc["above", "95% upper-bound"]))
    pooled_P     = float(cph.summary.loc["above", "p"])

    # Rows 2/3: subgroup × endpoint cross-arm KMs (4 panels)
    above_pfs = _km_pair(above, "pfs_t", "pfs_e")
    below_pfs = _km_pair(below, "pfs_t", "pfs_e")
    above_os  = _km_pair(above, "os_t",  "os_e")
    below_os  = _km_pair(below, "os_t",  "os_e")

    # Pack
    out = dict(
        # τ̂ histogram + threshold
        tau_arm1=arm1_tau, tau_arm0=arm0_tau, hist_bins=bins,
        hist_arm1=hist_arm1, hist_arm0=hist_arm0,
        tau0=tau0,
        # sliding window HR curve (empirical, kept for diagnostic plots)
        hr_curve_tau=hr_centers, hr_curve_HR=hr_vals,
        hr_curve_lo=hr_lo, hr_curve_hi=hr_hi,
        # Smooth analytic HR(τ̂) from the interaction Cox model - primary panel
        hr_smooth_tau=tau_grid, hr_smooth_HR=hr_smooth_HR,
        hr_smooth_lo=hr_smooth_lo, hr_smooth_hi=hr_smooth_hi,
        beta_T=beta_T, beta_int=beta_int,
        # pooled split by τ̂_0
        pooled_above_t=pooled_above["t"], pooled_above_s=pooled_above["s"],
        pooled_above_lo=pooled_above["s_lo"], pooled_above_hi=pooled_above["s_hi"],
        pooled_above_n=pooled_above["n"], pooled_above_ev=pooled_above["n_events"],
        pooled_below_t=pooled_below["t"], pooled_below_s=pooled_below["s"],
        pooled_below_lo=pooled_below["s_lo"], pooled_below_hi=pooled_below["s_hi"],
        pooled_below_n=pooled_below["n"], pooled_below_ev=pooled_below["n_events"],
        pooled_HR=pooled_HR, pooled_HR_lo=pooled_HR_lo, pooled_HR_hi=pooled_HR_hi,
        pooled_P=pooled_P, pooled_logrank_P=pooled_lr_p,
    )
    out.update(_flatten("above_pfs", above_pfs))
    out.update(_flatten("below_pfs", below_pfs))
    out.update(_flatten("above_os",  above_os))
    out.update(_flatten("below_os",  below_os))

    meta = dict(
        tau0=float(tau0),
        n_total=int(len(df)),
        n_arm1=int((df["arm"] == 1).sum()),
        n_arm0=int((df["arm"] == 0).sum()),
        n_above=int(len(above)),
        n_below=int(len(below)),
        arm1_label=cfg["arm1_label"],
        arm0_label=cfg["arm0_label"],
        pfs_xlim=cfg["pfs_xlim"],
        os_xlim=cfg["os_xlim"],
        interaction_HR=itx_pfs["HR"],
        interaction_HR_lo=itx_pfs["HR_lo"],
        interaction_HR_hi=itx_pfs["HR_hi"],
        interaction_P=itx_pfs["P"],
    )
    return out, meta


def compute_ablation(slices: dict, cfg: dict) -> dict:
    """For each feature slice, compute per-slice τ̂_0 and above/below PFS KMs."""
    out = {}
    for slice_name, csv_path in slices.items():
        if not csv_path.exists():
            print(f"  SKIP {slice_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        itx = interaction_test(df, df["tau"].values, time_col="pfs_t", event_col="pfs_e")
        tau0 = threshold_indifference(itx)
        df = _annotate(df, tau0)
        above = df[df["recommended"] == 1]
        below = df[df["recommended"] == 0]
        above_pfs = _km_pair(above, "pfs_t", "pfs_e")
        below_pfs = _km_pair(below, "pfs_t", "pfs_e")
        out[f"{slice_name}_tau0"] = float(tau0)
        out[f"{slice_name}_n_above"] = int(len(above))
        out[f"{slice_name}_n_below"] = int(len(below))
        out[f"{slice_name}_interaction_HR"] = itx["HR"]
        out[f"{slice_name}_interaction_P"] = itx["P"]
        out[f"{slice_name}_tau"] = df["tau"].values
        out.update(_flatten(f"{slice_name}_above", above_pfs))
        out.update(_flatten(f"{slice_name}_below", below_pfs))
    return out


def main():
    for name, cfg in COHORTS.items():
        if not cfg["pred_csv"].exists():
            print(f"[{name}] SKIP - {cfg['pred_csv']} not found")
            continue
        out_dir = RESULTS / "figures" / name / "figure6"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] loading {cfg['pred_csv']}")
        df = pd.read_csv(cfg["pred_csv"])
        panels, meta = compute_main(df, cfg)
        np.savez_compressed(out_dir / "panels.npz", **panels)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{name}] saved {out_dir/'panels.npz'}")
        print(f"[{name}] saved {out_dir/'meta.json'}")
        print(f"        τ̂_0={meta['tau0']:+.2f}  n_above={meta['n_above']}  n_below={meta['n_below']}")
        print(f"[{name}] computing ablation slices")
        abl = compute_ablation(cfg["ablation"], cfg)
        if abl:
            np.savez_compressed(out_dir / "ablation.npz", **abl)
            print(f"[{name}] saved {out_dir/'ablation.npz'}")
        else:
            print(f"[{name}] no ablation slices found")
        print()


if __name__ == "__main__":
    main()
