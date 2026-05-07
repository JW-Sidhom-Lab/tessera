"""Plot library for predictive-biomarker analyses.

Each function is self-contained: takes a DataFrame, label arguments for the
two arms, and an output stem; writes both PDF and PNG. Designed to work on
the per-patient predictions produced by `crc_discovery.py` / `pdac_discovery.py`.

Required columns in `df`:
  arm        : binary (1 = arm 1 / FOLFOX or FOLFIRINOX, 0 = arm 0)
  tau        : signed predicted treatment effect (sign-aligned)
  pfs_t, pfs_e, os_t, os_e
  recommended: 1 if tau > tau0 else 0  (added by driver)
  concordant : 1 if recommended == arm else 0  (added by driver)
  tertile    : 0 = top τ̂, 1 = mid, 2 = bottom  (added by driver)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test

from .plotutils import (
    ARM0_COLOR, ARM1_COLOR, TERT_COLORS, TERT_LABELS,
    ABOVE_COLOR, BELOW_COLOR, CONCORDANT_COLOR, DISCORDANT_COLOR,
    apply_default_style, format_p, format_hr, save_fig,
)


def _arm_hr(sub: pd.DataFrame, time_col: str, event_col: str) -> dict:
    """Cox HR for arm=1 vs arm=0 within a subgroup."""
    if sub["arm"].nunique() < 2 or len(sub) < 5:
        return {"HR": np.nan, "HR_lo": np.nan, "HR_hi": np.nan, "P": np.nan, "n": len(sub)}
    work = sub[[time_col, event_col, "arm"]].rename(columns={"arm": "T"})
    cph = CoxPHFitter().fit(work[[time_col, event_col, "T"]],
                              duration_col=time_col, event_col=event_col)
    HR    = float(np.exp(cph.params_["T"]))
    HR_lo = float(np.exp(cph.confidence_intervals_.loc["T", "95% lower-bound"]))
    HR_hi = float(np.exp(cph.confidence_intervals_.loc["T", "95% upper-bound"]))
    P     = float(cph.summary.loc["T", "p"])
    return {"HR": HR, "HR_lo": HR_lo, "HR_hi": HR_hi, "P": P, "n": len(sub)}


def _km_median_and_milestone(s: pd.Series, e: pd.Series, milestone: float) -> tuple[float, float]:
    km = KaplanMeierFitter().fit(s, e)
    med = km.median_survival_time_
    surv = float(km.predict(milestone))
    return float(med) if pd.notna(med) and np.isfinite(med) else np.nan, surv


# ============================================================================
# 1. Tau distribution (overall histogram with threshold marker)
# ============================================================================

def tau_distribution(df: pd.DataFrame, *, tau0: float, out_stem: Path,
                       cancer_label: str = "CRC",
                       arm1_label: str = "FOLFOX", arm0_label: str = "FOLFIRI"):
    """Overall τ̂ histogram with τ̂_0 threshold line and per-arm shading."""
    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(df["tau"].min(), df["tau"].max(), 60)
    arm1 = df[df["arm"] == 1]["tau"]
    arm0 = df[df["arm"] == 0]["tau"]
    ax.hist(arm0, bins=bins, alpha=0.55, color=ARM0_COLOR,
            label=f"{arm0_label} (n={len(arm0)})", edgecolor="white", linewidth=0.3)
    ax.hist(arm1, bins=bins, alpha=0.55, color=ARM1_COLOR,
            label=f"{arm1_label} (n={len(arm1)})", edgecolor="white", linewidth=0.3)
    ax.axvline(tau0, color="black", linestyle="--", linewidth=1.5,
                 label=f"τ̂_0 = {tau0:+.2f}")
    n_above = int((df["tau"] > tau0).sum())
    n_below = int((df["tau"] <= tau0).sum())
    ax.text(0.02, 0.98,
            f"Above τ̂_0 ({arm1_label}-favored):  n={n_above} ({100*n_above/len(df):.1f}%)\n"
            f"Below τ̂_0 ({arm0_label}-favored):  n={n_below} ({100*n_below/len(df):.1f}%)",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.4))
    ax.set_xlabel("Predicted treatment effect τ̂ (RMST units)")
    ax.set_ylabel("Patient count")
    ax.set_title(f"{cancer_label} - distribution of τ̂ with indifference threshold")
    ax.legend(loc="upper right", frameon=False)
    save_fig(fig, out_stem)


# ============================================================================
# 2. Tau density by arm (overlay) - should overlap if τ̂ isn't predicting arm
# ============================================================================

def tau_density_by_arm(df: pd.DataFrame, *, out_stem: Path,
                          cancer_label: str = "CRC",
                          arm1_label: str = "FOLFOX", arm0_label: str = "FOLFIRI"):
    """Overlapping density of τ̂ per arm; verifies τ̂ is not just an arm proxy."""
    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    arm1 = df[df["arm"] == 1]["tau"].values
    arm0 = df[df["arm"] == 0]["tau"].values
    bins = np.linspace(df["tau"].min(), df["tau"].max(), 50)
    ax.hist(arm0, bins=bins, density=True, color=ARM0_COLOR, alpha=0.4,
              edgecolor=ARM0_COLOR, label=f"{arm0_label} (n={len(arm0)})")
    ax.hist(arm1, bins=bins, density=True, color=ARM1_COLOR, alpha=0.4,
              edgecolor=ARM1_COLOR, label=f"{arm1_label} (n={len(arm1)})")
    # Annotate medians
    ax.axvline(np.median(arm1), color=ARM1_COLOR, linestyle=":", linewidth=1.2)
    ax.axvline(np.median(arm0), color=ARM0_COLOR, linestyle=":", linewidth=1.2)
    from scipy.stats import mannwhitneyu, ks_2samp
    u, p_mw = mannwhitneyu(arm1, arm0, alternative="two-sided")
    _, p_ks = ks_2samp(arm1, arm0)
    ax.text(0.02, 0.98,
            f"Mann-Whitney P = {format_p(p_mw)}\n"
            f"KS P = {format_p(p_ks)}\n"
            f"median {arm1_label}: {np.median(arm1):+.2f}\n"
            f"median {arm0_label}: {np.median(arm0):+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.4))
    ax.set_xlabel("τ̂")
    ax.set_ylabel("Density")
    ax.set_title(f"{cancer_label} - τ̂ density by treatment arm "
                  f"(should overlap if τ̂ is not an arm proxy)")
    ax.legend(loc="upper right", frameon=False)
    save_fig(fig, out_stem)


# ============================================================================
# 3. Personalized HR curve - sliding-window HR(arm1 vs arm0) along τ̂ percentile
# ============================================================================

def personalized_hr_curve(df: pd.DataFrame, *, time_col: str, event_col: str,
                              tau0: float, out_stem: Path,
                              window_frac: float = 0.30, n_points: int = 50,
                              cancer_label: str = "CRC",
                              endpoint: str = "PFS",
                              arm1_label: str = "FOLFOX",
                              arm0_label: str = "FOLFIRI"):
    """At each percentile of τ̂, fit a Cox in a sliding window centered on
    that percentile and plot HR(arm1 vs arm0) with CI ribbon. The threshold
    crossing is where the HR curve = 1."""
    apply_default_style()
    df = df.sort_values("tau").reset_index(drop=True)
    n = len(df)
    win = int(window_frac * n)
    if win < 30:
        win = 30
    points = np.linspace(win // 2, n - win // 2 - 1, n_points).astype(int)

    HRs, HR_los, HR_his, taus = [], [], [], []
    for i in points:
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2)
        sub = df.iloc[lo:hi]
        h = _arm_hr(sub, time_col, event_col)
        HRs.append(h["HR"]); HR_los.append(h["HR_lo"]); HR_his.append(h["HR_hi"])
        taus.append(float(sub["tau"].median()))

    fig, ax = plt.subplots(figsize=(9, 5))
    HRs    = np.array(HRs)
    HR_los = np.array(HR_los)
    HR_his = np.array(HR_his)
    taus   = np.array(taus)

    ax.fill_between(taus, HR_los, HR_his, alpha=0.20, color="#5b8a72")
    ax.plot(taus, HRs, color="#1b4332", linewidth=2.0,
              label=f"HR({arm1_label} vs {arm0_label}) on {endpoint}")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    ax.axvline(tau0, color="black", linestyle="--", linewidth=1.2,
                 label=f"τ̂_0 = {tau0:+.2f}")

    # Rug at bottom for individual τ̂ values, colored by arm
    yrug = ax.get_ylim()[0]
    arm1 = df[df["arm"] == 1]["tau"]
    arm0 = df[df["arm"] == 0]["tau"]
    ax.scatter(arm1, np.full(len(arm1), yrug), s=4, color=ARM1_COLOR, alpha=0.4, marker="|")
    ax.scatter(arm0, np.full(len(arm0), yrug * 0.95), s=4, color=ARM0_COLOR, alpha=0.4, marker="|")

    ax.set_yscale("log")
    ax.set_xlabel("τ̂  (sliding-window center)")
    ax.set_ylabel(f"HR({arm1_label} vs {arm0_label})  [log scale]")
    ax.set_title(f"{cancer_label} - personalized HR curve  ({endpoint}, "
                  f"window = {int(window_frac*100)}% of cohort)")
    ax.legend(loc="best", frameon=False)
    save_fig(fig, out_stem)


# ============================================================================
# 4. Threshold-split cross-arm KM (above vs below threshold, per endpoint)
# ============================================================================

def threshold_split_km(df: pd.DataFrame, *, time_col: str, event_col: str,
                          out_stem: Path, endpoint: str = "PFS",
                          cancer_label: str = "CRC",
                          arm1_label: str = "FOLFOX",
                          arm0_label: str = "FOLFIRI", xlim: float = None):
    """Two-panel: ABOVE τ̂_0 cross-arm KM | BELOW τ̂_0 cross-arm KM."""
    apply_default_style()
    above = df["recommended"] == 1
    below = df["recommended"] == 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, lbl, mask, badge_color in [
        (axes[0], f"ABOVE τ̂_0 ({arm1_label}-favored)", above, ABOVE_COLOR),
        (axes[1], f"BELOW τ̂_0 ({arm0_label}-favored)", below, BELOW_COLOR),
    ]:
        sub = df[mask]
        for arm_val, color, alab in [(1, ARM1_COLOR, arm1_label), (0, ARM0_COLOR, arm0_label)]:
            s = sub[sub["arm"] == arm_val]
            if len(s) < 3: continue
            km = KaplanMeierFitter().fit(s[time_col], s[event_col],
                label=f"{alab} n={len(s)}, ev={int(s[event_col].sum())}")
            km.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.10,
                                          color=color, linewidth=1.6)
        h = _arm_hr(sub, time_col, event_col)
        lr = logrank_test(sub[sub["arm"]==1][time_col], sub[sub["arm"]==0][time_col],
                            event_observed_A=sub[sub["arm"]==1][event_col],
                            event_observed_B=sub[sub["arm"]==0][event_col])
        ax.text(0.02, 0.05,
                f"HR({arm1_label} vs {arm0_label}) = {format_hr(h['HR'], h['HR_lo'], h['HR_hi'])}\n"
                f"P = {format_p(h['P'])}   logrank P = {format_p(float(lr.p_value))}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor=badge_color, linewidth=0.6))
        if xlim is None:
            xlim_use = sub[time_col].max() * 1.02
        else:
            xlim_use = xlim
        ax.set_xlim(0, xlim_use); ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"{endpoint} (months)")
        ax.set_ylabel("Survival probability")
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        ax.set_title(f"{lbl}   n={len(sub)}")
    fig.suptitle(f"{cancer_label} - {endpoint} cross-arm KM split at τ̂_0", fontsize=12, y=1.00)
    fig.tight_layout()
    save_fig(fig, out_stem)


# ============================================================================
# 5. Within-arm tertile KM (within one arm, KM by τ̂ tertile)
# ============================================================================

def within_arm_tertile_km(df: pd.DataFrame, *, arm_value: int,
                              time_col: str, event_col: str,
                              out_stem: Path, endpoint: str = "PFS",
                              cancer_label: str = "CRC",
                              arm_label: str = "FOLFOX", xlim: float = None):
    """Within a single arm, KM by τ̂ tertile."""
    apply_default_style()
    sub = df[df["arm"] == arm_value].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for q, lbl, color in zip([0, 1, 2], TERT_LABELS, TERT_COLORS):
        s = sub[sub["tertile"] == q]
        if len(s) < 3: continue
        km = KaplanMeierFitter().fit(s[time_col], s[event_col],
            label=f"{lbl}  n={len(s)}, ev={int(s[event_col].sum())}")
        km.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.10,
                                      color=color, linewidth=1.6)
    cph = CoxPHFitter(penalizer=0.001).fit(
        sub[[time_col, event_col, "tau"]], duration_col=time_col, event_col=event_col)
    HR = float(np.exp(cph.params_["tau"] * sub["tau"].std()))
    HR_lo = float(np.exp(cph.confidence_intervals_.loc["tau", "95% lower-bound"] * sub["tau"].std()))
    HR_hi = float(np.exp(cph.confidence_intervals_.loc["tau", "95% upper-bound"] * sub["tau"].std()))
    P = float(cph.summary.loc["tau", "p"])
    mv = multivariate_logrank_test(sub[time_col], sub["tertile"], sub[event_col])
    ax.text(0.02, 0.05,
            f"Cox HR per +1 SD τ̂ = {format_hr(HR, HR_lo, HR_hi)}   P = {format_p(P)}\n"
            f"3-way logrank P = {format_p(float(mv.p_value))}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.4))
    if xlim is None:
        xlim = sub[time_col].max() * 1.02
    ax.set_xlim(0, xlim); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{endpoint} (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"{cancer_label} - {endpoint} within {arm_label} arm by τ̂ tertile  (n={len(sub)})")
    save_fig(fig, out_stem)


# ============================================================================
# 6. Concordance KM - concordant (got recommended) vs discordant
# ============================================================================

def concordance_km(df: pd.DataFrame, *, time_col: str, event_col: str,
                       out_stem: Path, endpoint: str = "PFS",
                       cancer_label: str = "CRC", xlim: float = None):
    """KM for patients whose actual treatment matched the biomarker recommendation
    vs those where they didn't."""
    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for cval, lbl, color in [(1, "Concordant (got recommended)", CONCORDANT_COLOR),
                                (0, "Discordant (got opposite)", DISCORDANT_COLOR)]:
        s = df[df["concordant"] == cval]
        if len(s) < 3: continue
        km = KaplanMeierFitter().fit(s[time_col], s[event_col],
            label=f"{lbl}  n={len(s)}, ev={int(s[event_col].sum())}")
        km.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.10,
                                      color=color, linewidth=1.7)
    sub = df[[time_col, event_col, "concordant"]].copy()
    cph = CoxPHFitter().fit(sub.rename(columns={"concordant":"C"})[[time_col, event_col, "C"]],
                              duration_col=time_col, event_col=event_col)
    HR = float(np.exp(cph.params_["C"]))
    HR_lo = float(np.exp(cph.confidence_intervals_.loc["C", "95% lower-bound"]))
    HR_hi = float(np.exp(cph.confidence_intervals_.loc["C", "95% upper-bound"]))
    P = float(cph.summary.loc["C", "p"])
    lr = logrank_test(df[df["concordant"]==1][time_col], df[df["concordant"]==0][time_col],
                        event_observed_A=df[df["concordant"]==1][event_col],
                        event_observed_B=df[df["concordant"]==0][event_col])
    ax.text(0.02, 0.05,
            f"HR(concordant vs discordant) = {format_hr(HR, HR_lo, HR_hi)}   P = {format_p(P)}\n"
            f"logrank P = {format_p(float(lr.p_value))}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.4))
    if xlim is None:
        xlim = df[time_col].max() * 1.02
    ax.set_xlim(0, xlim); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{endpoint} (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"{cancer_label} - {endpoint} by concordance with biomarker recommendation")
    save_fig(fig, out_stem)


# ============================================================================
# 7. Four-quadrant KM - (Rec arm × Got arm) full grid
# ============================================================================

def four_quadrant_km(df: pd.DataFrame, *, time_col: str, event_col: str,
                        out_stem: Path, endpoint: str = "PFS",
                        cancer_label: str = "CRC",
                        arm1_label: str = "FOLFOX",
                        arm0_label: str = "FOLFIRI", xlim: float = None):
    """2×2 grid of KMs for each (Rec, Got) cell."""
    apply_default_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True, sharex=True)
    cells = [
        (axes[0, 0], 1, 1, f"Rec {arm1_label} + Got {arm1_label} (concordant)", ARM1_COLOR),
        (axes[0, 1], 1, 0, f"Rec {arm1_label} + Got {arm0_label} (discordant)", DISCORDANT_COLOR),
        (axes[1, 0], 0, 1, f"Rec {arm0_label} + Got {arm1_label} (discordant)", DISCORDANT_COLOR),
        (axes[1, 1], 0, 0, f"Rec {arm0_label} + Got {arm0_label} (concordant)", ARM0_COLOR),
    ]
    for ax, rec, got, lbl, color in cells:
        s = df[(df["recommended"] == rec) & (df["arm"] == got)]
        ax.set_title(f"{lbl}\nn={len(s)}, ev={int(s[event_col].sum())}", fontsize=10)
        if len(s) < 3:
            ax.text(0.5, 0.5, "n too small", ha="center", va="center", transform=ax.transAxes)
            continue
        km = KaplanMeierFitter().fit(s[time_col], s[event_col], label=lbl)
        km.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15,
                                      color=color, linewidth=1.6)
        med = km.median_survival_time_
        med_str = f"{med:.1f}" if pd.notna(med) and np.isfinite(med) else "not reached"
        m12 = float(km.predict(min(12.0, s[time_col].max())))
        ax.text(0.02, 0.05,
                f"Median {endpoint} = {med_str}\n12-mo {endpoint}-free = {100*m12:.1f}%",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.4))
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    if xlim is None:
        xlim = df[time_col].max() * 1.02
    for ax in axes.ravel():
        ax.set_xlim(0, xlim); ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"{endpoint} (months)")
        ax.set_ylabel("Survival probability")
    fig.suptitle(f"{cancer_label} - {endpoint} by (recommended × actual) treatment quadrants",
                  fontsize=12, y=1.00)
    fig.tight_layout()
    save_fig(fig, out_stem)


# ============================================================================
# 8. Prognostic-test 2×3 grid (validates that τ̂ is predictive, not prognostic)
# ============================================================================

def prognostic_test_grid(df: pd.DataFrame, *, out_stem: Path,
                              cancer_label: str = "CRC",
                              arm1_label: str = "FOLFOX",
                              arm0_label: str = "FOLFIRI",
                              pfs_xlim: float = 24.0, os_xlim: float = 60.0):
    """2 rows (PFS, OS) × 3 cols (pooled, arm1-only, arm0-only) of KMs by τ̂ tertile.

    If τ̂ is purely predictive: pooled column has overlapping curves; per-arm columns
    fan out in OPPOSITE directions.
    """
    apply_default_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")
    cohorts = [("Pooled", df),
               (f"{arm1_label} arm only", df[df["arm"] == 1]),
               (f"{arm0_label} arm only", df[df["arm"] == 0])]
    rows = [("PFS", "pfs_t", "pfs_e", pfs_xlim),
            ("OS",  "os_t",  "os_e",  os_xlim)]
    for r, (ep, T_col, E_col, xlim) in enumerate(rows):
        for c, (chrt_lbl, chrt) in enumerate(cohorts):
            ax = axes[r, c]
            for q, lbl, color in zip([0, 1, 2], TERT_LABELS, TERT_COLORS):
                s = chrt[chrt["tertile"] == q]
                if len(s) < 3: continue
                km = KaplanMeierFitter().fit(s[T_col], s[E_col],
                    label=f"{lbl} n={len(s)}, ev={int(s[E_col].sum())}")
                km.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.10,
                                              color=color, linewidth=1.4)
            sub = chrt[[T_col, E_col, "tau"]].copy()
            cph = CoxPHFitter(penalizer=0.001).fit(sub, duration_col=T_col, event_col=E_col)
            HR = float(np.exp(cph.params_["tau"] * sub["tau"].std()))
            HR_lo = float(np.exp(cph.confidence_intervals_.loc["tau", "95% lower-bound"] * sub["tau"].std()))
            HR_hi = float(np.exp(cph.confidence_intervals_.loc["tau", "95% upper-bound"] * sub["tau"].std()))
            P = float(cph.summary.loc["tau", "p"])
            mv = multivariate_logrank_test(chrt[T_col], chrt["tertile"], chrt[E_col])
            ax.text(0.02, 0.05,
                    f"Cox HR per +1 SD = {HR:.2f} [{HR_lo:.2f}, {HR_hi:.2f}]  P = {format_p(P)}\n"
                    f"3-way logrank P = {format_p(float(mv.p_value))}",
                    transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.88, edgecolor="gray", linewidth=0.4))
            ax.set_xlim(0, xlim); ax.set_ylim(0, 1.02)
            ax.set_xlabel(f"{ep} (months)")
            if c == 0: ax.set_ylabel(f"{ep} survival probability")
            ax.legend(loc="lower left", frameon=False, fontsize=8)
            ax.set_title(f"{ep} - {chrt_lbl} (n={len(chrt)})")
    fig.suptitle(f"{cancer_label} - τ̂ should be predictive (per-arm fans opposite), "
                  f"not prognostic (pooled curves overlap)", fontsize=11, y=1.00)
    fig.tight_layout()
    save_fig(fig, out_stem)


# ============================================================================
# 9. Forest plot (generic - takes list of subgroup HRs)
# ============================================================================

def forest_plot(rows: list[dict], *, out_stem: Path, title: str,
                  ref_line: float = 1.0, xlim_log: tuple = (0.3, 3.0)):
    """Generic forest plot, single-axis design.

    Layout (left to right): subgroup label | n | forest markers | HR text.
    Each row = dict(label, n, HR, HR_lo, HR_hi, P).
    Rows where HR is NaN render as a separator (label-only).
    """
    apply_default_style()
    n_rows = len(rows)
    fig_h = max(3.5, 0.45 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    # Coordinate system: x in HR space (log), y = row index (top is row 0)
    ax.set_xscale("log")
    ax.set_xlim(*xlim_log)
    ax.set_ylim(n_rows - 0.5, -0.5)   # invert so row 0 is at top
    ax.axvline(ref_line, color="gray", linestyle=":", linewidth=1.0)

    # Column x positions in axes-fraction coordinates for the text tracks.
    # We keep the forest plot itself in the central data-coords region;
    # labels and HR text live to the LEFT and RIGHT of that region using
    # blended (axes, data) transforms.
    label_x = -0.32   # axes-fraction x for subgroup label column
    n_x     = -0.06   # axes-fraction x for n column
    hr_x    = +1.04   # axes-fraction x for HR-text column
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for i, r in enumerate(rows):
        # Header rows (NaN HR) are typeset bold-ish as section dividers
        is_header = not np.isfinite(r.get("HR", np.nan))
        weight = "bold" if is_header else "normal"
        color  = "#222222" if is_header else "black"
        ax.text(label_x, i, str(r["label"]), transform=trans,
                  va="center", ha="left", fontsize=9, color=color, fontweight=weight)
        if not is_header:
            ax.text(n_x, i, f"n={r['n']}", transform=trans,
                      va="center", ha="right", fontsize=8, color="gray")
            # forest segment + marker
            ax.plot([r["HR_lo"], r["HR_hi"]], [i, i],
                      color="black", linewidth=1.3, zorder=2)
            ax.scatter([r["HR"]], [i], s=44, color="#1b4332", zorder=3)
            # right-side numbers
            txt = (f"{r['HR']:.2f} [{r['HR_lo']:.2f}, {r['HR_hi']:.2f}]   "
                   f"P={format_p(r['P'])}")
            ax.text(hr_x, i, txt, transform=trans,
                      va="center", ha="left", fontsize=9, family="monospace")

    # Tidy axis: no y ticks (rows are labeled by hand), keep x ticks for HR scale
    ax.set_yticks([])
    ax.set_xlabel("Hazard ratio (log scale)")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", pad=14)

    # Reserve room on left/right for the text tracks
    fig.subplots_adjust(left=0.30, right=0.70, top=0.92, bottom=0.10)
    save_fig(fig, out_stem)

# Need access to matplotlib.transforms inside forest_plot - import lazily here.
import matplotlib as mpl  # noqa: E402


# ============================================================================
# 10. Tau by subgroup (categorical demographic)
# ============================================================================

def tau_by_subgroup(df: pd.DataFrame, *, subgroup_col: str, out_stem: Path,
                       cancer_label: str = "CRC", title_suffix: str = ""):
    """Boxplot/strip of τ̂ by a categorical subgroup column."""
    apply_default_style()
    work = df.dropna(subset=["tau", subgroup_col]).copy()
    cats = sorted(work[subgroup_col].unique())
    data = [work[work[subgroup_col] == c]["tau"].values for c in cats]
    counts = [len(d) for d in data]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cats)), 5))
    bp = ax.boxplot(data, positions=range(len(cats)), widths=0.55, patch_artist=True,
                      medianprops=dict(color="black"), showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a8c5d8"); patch.set_alpha(0.8)
    # add jittered points
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        if len(d) == 0: continue
        x = rng.normal(i, 0.06, len(d))
        ax.scatter(x, d, s=8, color="#1b4332", alpha=0.4)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(cats, counts)], fontsize=9)
    ax.set_ylabel("τ̂")
    ax.set_title(f"{cancer_label} - τ̂ by {subgroup_col} {title_suffix}".rstrip())
    save_fig(fig, out_stem)


# ============================================================================
# 11. Clinical benefit bars - median PFS/OS per (arm × τ̂ group)
# ============================================================================

def clinical_benefit_bars(df: pd.DataFrame, *, group_col: str, group_labels: list[str],
                              out_stem: Path, cancer_label: str = "CRC",
                              arm1_label: str = "FOLFOX",
                              arm0_label: str = "FOLFIRI"):
    """Grouped bar chart: median PFS and median OS per (arm × group)."""
    apply_default_style()
    # Compute medians
    rows = []
    for g_val, g_lbl in enumerate(group_labels):
        for arm_val, arm_lbl in [(1, arm1_label), (0, arm0_label)]:
            s = df[(df[group_col] == g_val) & (df["arm"] == arm_val)]
            if len(s) < 5:
                rows.append({"group": g_lbl, "arm": arm_lbl, "med_pfs": np.nan,
                             "med_os": np.nan, "n": len(s)})
                continue
            pfs_med, _ = _km_median_and_milestone(s["pfs_t"], s["pfs_e"], milestone=12)
            os_med,  _ = _km_median_and_milestone(s["os_t"],  s["os_e"],  milestone=36)
            rows.append({"group": g_lbl, "arm": arm_lbl,
                          "med_pfs": pfs_med, "med_os": os_med, "n": len(s)})
    bench = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ep in [(axes[0], "med_pfs", "PFS"), (axes[1], "med_os", "OS")]:
        x = np.arange(len(group_labels))
        w = 0.36
        v_arm1 = [bench[(bench["group"]==g) & (bench["arm"]==arm1_label)][key].values[0]
                    for g in group_labels]
        v_arm0 = [bench[(bench["group"]==g) & (bench["arm"]==arm0_label)][key].values[0]
                    for g in group_labels]
        n_arm1 = [int(bench[(bench["group"]==g) & (bench["arm"]==arm1_label)]["n"].values[0])
                    for g in group_labels]
        n_arm0 = [int(bench[(bench["group"]==g) & (bench["arm"]==arm0_label)]["n"].values[0])
                    for g in group_labels]
        bars1 = ax.bar(x - w/2, v_arm1, w, color=ARM1_COLOR, label=arm1_label, alpha=0.85)
        bars0 = ax.bar(x + w/2, v_arm0, w, color=ARM0_COLOR, label=arm0_label, alpha=0.85)
        for b, v, n in zip(bars1, v_arm1, n_arm1):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}\n(n={n})",
                          ha="center", fontsize=8)
        for b, v, n in zip(bars0, v_arm0, n_arm0):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}\n(n={n})",
                          ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(group_labels)
        ax.set_ylabel(f"Median {ep} (months)")
        ax.set_title(f"Median {ep} per ({group_col} × arm)")
        ax.legend(loc="upper right", frameon=False)
    fig.suptitle(f"{cancer_label} - clinical benefit by τ̂ subgroup", fontsize=12, y=1.00)
    fig.tight_layout()
    save_fig(fig, out_stem)


# ============================================================================
# 12. Headline composite (multi-panel publication figure)
# ============================================================================

def headline_composite(df: pd.DataFrame, *, tau0: float, out_stem: Path,
                          cancer_label: str = "CRC",
                          arm1_label: str = "FOLFOX",
                          arm0_label: str = "FOLFIRI",
                          pfs_xlim: float = 24, os_xlim: float = 60):
    """6-panel composite:
       a) τ̂ histogram with threshold
       b) Personalized HR curve
       c) ABOVE τ̂_0 cross-arm KM (PFS)
       d) BELOW τ̂_0 cross-arm KM (PFS)
       e) ABOVE τ̂_0 cross-arm KM (OS)
       f) BELOW τ̂_0 cross-arm KM (OS)
    """
    apply_default_style()
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.45, wspace=0.25)

    # Panel a: τ̂ histogram
    ax_a = fig.add_subplot(gs[0, 0])
    bins = np.linspace(df["tau"].min(), df["tau"].max(), 50)
    arm1 = df[df["arm"] == 1]["tau"]; arm0 = df[df["arm"] == 0]["tau"]
    ax_a.hist(arm0, bins=bins, alpha=0.55, color=ARM0_COLOR,
                label=f"{arm0_label} (n={len(arm0)})", edgecolor="white", linewidth=0.3)
    ax_a.hist(arm1, bins=bins, alpha=0.55, color=ARM1_COLOR,
                label=f"{arm1_label} (n={len(arm1)})", edgecolor="white", linewidth=0.3)
    ax_a.axvline(tau0, color="black", linestyle="--", linewidth=1.4,
                   label=f"τ̂_0 = {tau0:+.2f}")
    ax_a.set_xlabel("τ̂"); ax_a.set_ylabel("Patients")
    ax_a.set_title("a)  τ̂ distribution with indifference threshold", loc="left")
    ax_a.legend(loc="upper right", frameon=False, fontsize=8)

    # Panel b: personalized HR curve (PFS)
    ax_b = fig.add_subplot(gs[0, 1])
    df_s = df.sort_values("tau").reset_index(drop=True)
    n = len(df_s); win = max(30, int(0.30 * n))
    points = np.linspace(win // 2, n - win // 2 - 1, 50).astype(int)
    HRs, los, his, taus = [], [], [], []
    for i in points:
        sub = df_s.iloc[max(0, i - win // 2):min(n, i + win // 2)]
        h = _arm_hr(sub, "pfs_t", "pfs_e")
        HRs.append(h["HR"]); los.append(h["HR_lo"]); his.append(h["HR_hi"])
        taus.append(float(sub["tau"].median()))
    HRs, los, his, taus = map(np.array, (HRs, los, his, taus))
    ax_b.fill_between(taus, los, his, alpha=0.25, color="#5b8a72")
    ax_b.plot(taus, HRs, color="#1b4332", linewidth=2)
    ax_b.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    ax_b.axvline(tau0, color="black", linestyle="--", linewidth=1.2)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("τ̂  (window-median)")
    ax_b.set_ylabel(f"HR({arm1_label} vs {arm0_label}) [log]")
    ax_b.set_title(f"b)  Personalized HR curve - PFS", loc="left")

    # Panels c–f: threshold-split KMs for PFS and OS
    panels = [
        ("c", gs[1, 0], "PFS", "pfs_t", "pfs_e", pfs_xlim, df["recommended"]==1, ABOVE_COLOR, "Above τ̂_0"),
        ("d", gs[1, 1], "PFS", "pfs_t", "pfs_e", pfs_xlim, df["recommended"]==0, BELOW_COLOR, "Below τ̂_0"),
        ("e", gs[2, 0], "OS",  "os_t",  "os_e",  os_xlim,  df["recommended"]==1, ABOVE_COLOR, "Above τ̂_0"),
        ("f", gs[2, 1], "OS",  "os_t",  "os_e",  os_xlim,  df["recommended"]==0, BELOW_COLOR, "Below τ̂_0"),
    ]
    for letter, slot, ep, T, E, xl, mask, color, badge in panels:
        ax = fig.add_subplot(slot)
        sub = df[mask]
        for arm_val, c, alab in [(1, ARM1_COLOR, arm1_label), (0, ARM0_COLOR, arm0_label)]:
            s = sub[sub["arm"] == arm_val]
            if len(s) < 3: continue
            km = KaplanMeierFitter().fit(s[T], s[E],
                label=f"{alab} n={len(s)}")
            km.plot_survival_function(ax=ax, ci_show=False, color=c, linewidth=1.6)
        h = _arm_hr(sub, T, E)
        ax.text(0.02, 0.05,
                f"HR = {format_hr(h['HR'], h['HR_lo'], h['HR_hi'])}\nP = {format_p(h['P'])}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor=color, linewidth=0.6))
        ax.set_xlim(0, xl); ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"{ep} (months)"); ax.set_ylabel("Survival probability")
        ax.legend(loc="lower left", frameon=False, fontsize=8)
        ax.set_title(f"{letter})  {badge} - {ep}  (n={len(sub)})", loc="left")

    fig.suptitle(f"{cancer_label} - predictive biomarker headline ({arm1_label} vs {arm0_label})",
                  fontsize=13, y=0.995)
    save_fig(fig, out_stem)
