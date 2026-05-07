"""Per-signature predictive-biomarker validation for PDAC PMD decomposition.

For each PMD signature k we compute a per-patient score
    s_pk = u_pk × D_k × sum_g v_gk
which is the patient's contribution to τ̂_p from signature k. We then run the
same interaction-Cox + indifference-threshold + per-stratum analysis as the
main pipeline, INDIVIDUALLY for each signature.

A signature passes the predictive-biomarker test if:
  - PFS interaction P < 0.05 (treatment-by-signature interaction is real)
  - Above-τ̂0 stratum and below-τ̂0 stratum have direction-reversed arm HRs
  - Both per-stratum HRs are < 1 / > 1 with reasonable confidence

PDAC arms: FOLFIRINOX (arm=1) vs Gem/Abraxane (arm=0).
HR<1 means FOLFIRINOX-favored; HR>1 means GA-favored.

Outputs:
  attribution_analysis/pdac_signatures/signature_validation/
    signature_metrics.tsv  - interaction HR/P, τ̂0, per-stratum HR/P/n
    fig_signature_KMs.pdf  - multi-panel KM grid (one row per signature)
"""
from __future__ import annotations

import sys
import pickle
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from core.dr import (
    interaction_test,
    threshold_indifference,
    per_group_arm_hr,
)
from core.arm_mapping import label_signature_feature


OUT_DIR = ROOT / "attribution_analysis" / "pdac_signatures"
VAL_DIR = OUT_DIR / "signature_validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)


def signature_score(U: np.ndarray, D: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    """Per-patient contribution to τ̂ from signature k (same units as τ̂)."""
    return U[:, k] * D[k] * V[:, k].sum()


def _safe_arm_hr(df, mask, *, time_col, event_col):
    """NaN-filled wrapper for degenerate strata."""
    sub = df[mask]
    nan_out = {"n": int(len(sub)), "HR": float("nan"), "HR_lo": float("nan"),
                  "HR_hi": float("nan"), "P": float("nan")}
    if len(sub) < 10:
        return nan_out
    arm_counts = sub["arm"].value_counts()
    if (arm_counts < 5).any() or len(arm_counts) < 2:
        return nan_out
    if sub[event_col].sum() < 3:
        return nan_out
    try:
        return per_group_arm_hr(df, mask, time_col=time_col, event_col=event_col)
    except Exception as e:
        print(f"    [warn] arm-HR fit failed on stratum (n={len(sub)}): {type(e).__name__}: {e}")
        return nan_out


def validate_signature(df: pd.DataFrame, score: np.ndarray, *, label: str) -> dict:
    work = df.copy()
    raw_itx = interaction_test(work, score, time_col="pfs_t", event_col="pfs_e")
    sign = -1.0 if raw_itx["HR"] > 1.0 else +1.0
    s = sign * score

    itx_pfs = interaction_test(work, s, time_col="pfs_t", event_col="pfs_e")
    itx_os  = interaction_test(work, s, time_col="os_t",  event_col="os_e")
    tau0 = float(threshold_indifference(itx_pfs))
    above = s > tau0
    below = ~above

    a_pfs = _safe_arm_hr(work, above, time_col="pfs_t", event_col="pfs_e")
    b_pfs = _safe_arm_hr(work, below, time_col="pfs_t", event_col="pfs_e")
    a_os  = _safe_arm_hr(work, above, time_col="os_t",  event_col="os_e")
    b_os  = _safe_arm_hr(work, below, time_col="os_t",  event_col="os_e")

    return dict(
        label=label,
        sign=sign,
        score=s,
        itx_pfs_HR=itx_pfs["HR"], itx_pfs_HR_lo=itx_pfs["HR_lo"],
        itx_pfs_HR_hi=itx_pfs["HR_hi"], itx_pfs_P=itx_pfs["P"],
        itx_os_HR=itx_os["HR"], itx_os_P=itx_os["P"],
        tau0=tau0,
        above_n=int(above.sum()),
        below_n=int(below.sum()),
        above_pfs_HR=a_pfs["HR"], above_pfs_HR_lo=a_pfs["HR_lo"],
        above_pfs_HR_hi=a_pfs["HR_hi"], above_pfs_P=a_pfs["P"],
        below_pfs_HR=b_pfs["HR"], below_pfs_HR_lo=b_pfs["HR_lo"],
        below_pfs_HR_hi=b_pfs["HR_hi"], below_pfs_P=b_pfs["P"],
        above_os_HR=a_os["HR"], above_os_P=a_os["P"],
        below_os_HR=b_os["HR"], below_os_P=b_os["P"],
        above=above,
        below=below,
    )


def km_panel(ax, df, mask, *, time_col, event_col, label, hr, p, n,
              arm_colors=("#c0392b", "#2980b9"),
              arm_labels=("FOLFIRINOX", "Gem/Abraxane"),
              xlim=24.0):
    sub = df[mask]
    for arm_v, color, alabel in zip([1, 0], arm_colors, arm_labels):
        s = sub[sub["arm"] == arm_v]
        if len(s) < 2:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(s[time_col].values, event_observed=s[event_col].values, label=alabel)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, lw=1.5)
    ax.set_xlim(0, xlim); ax.set_ylim(0, 1.02)
    ax.set_title(f"{label}  n={n}\nHR={hr:.2f}  P={p:.2e}", fontsize=8)
    ax.set_xlabel("PFS (months)", fontsize=8)
    ax.set_ylabel("Survival", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, frameon=False)


def main():
    with open(OUT_DIR / "attribution_matrix.pkl", "rb") as fh:
        state = pickle.load(fh)
    df = state["df"]
    A = state["A"]
    print(f"loaded attribution matrix: A={A.shape}, df={df.shape}", flush=True)

    K_choice = sys.argv[1] if len(sys.argv) > 1 else "5"
    pmd_files = sorted(glob.glob(str(OUT_DIR / f"pmd_K{K_choice}_cv*.pkl")))
    if not pmd_files:
        raise FileNotFoundError(
            f"No PMD fit found for K={K_choice} in {OUT_DIR}.")
    pmd_path = pmd_files[0]
    with open(pmd_path, "rb") as fh:
        pmd = pickle.load(fh)
    res = pmd["res"]
    K = res["K"]
    print(f"loaded PMD fit: {Path(pmd_path).name}, K={K}, "
          f"expl_var={res['explained_var']:.3f}", flush=True)

    is_arm = state["is_arm"]
    arm_dir = state.get("arm_dir")
    features = state["features"]
    rows = []
    sig_results = []
    for k in range(K):
        s_raw = signature_score(res["U"], res["D"], res["V"], k)
        v = res["V"][:, k]
        order = np.argsort(-np.abs(v))
        top_lab = "  ".join(
            label_signature_feature(features[i], float(v[i]), bool(is_arm[i]), arm_dir)
            for i in order[:6] if abs(v[i]) > 1e-10
        )
        print(f"\n--- Signature {k}: {top_lab} ---", flush=True)
        out = validate_signature(df, s_raw, label=f"sig{k} [{top_lab}]")
        sig_results.append(out)
        rows.append({
            "sig": k,
            "top_genes": top_lab,
            "sign_flip": out["sign"],
            "interaction_PFS_HR": out["itx_pfs_HR"],
            "interaction_PFS_HR_lo": out["itx_pfs_HR_lo"],
            "interaction_PFS_HR_hi": out["itx_pfs_HR_hi"],
            "interaction_PFS_P": out["itx_pfs_P"],
            "interaction_OS_HR": out["itx_os_HR"],
            "interaction_OS_P": out["itx_os_P"],
            "tau0": out["tau0"],
            "above_n": out["above_n"],
            "below_n": out["below_n"],
            "above_PFS_HR": out["above_pfs_HR"],
            "above_PFS_HR_lo": out["above_pfs_HR_lo"],
            "above_PFS_HR_hi": out["above_pfs_HR_hi"],
            "above_PFS_P": out["above_pfs_P"],
            "below_PFS_HR": out["below_pfs_HR"],
            "below_PFS_HR_lo": out["below_pfs_HR_lo"],
            "below_PFS_HR_hi": out["below_pfs_HR_hi"],
            "below_PFS_P": out["below_pfs_P"],
            "above_OS_HR": out["above_os_HR"],
            "above_OS_P": out["above_os_P"],
            "below_OS_HR": out["below_os_HR"],
            "below_OS_P": out["below_os_P"],
        })

        print(f"\n=== Signature {k} ===")
        print(f"  PFS interaction HR={out['itx_pfs_HR']:.3f} "
              f"[{out['itx_pfs_HR_lo']:.3f}, {out['itx_pfs_HR_hi']:.3f}]  "
              f"P={out['itx_pfs_P']:.2e}")
        print(f"  τ̂_0 = {out['tau0']:+.2f}")
        print(f"  ABOVE n={out['above_n']:4d}  PFS HR={out['above_pfs_HR']:.3f} "
              f"[{out['above_pfs_HR_lo']:.3f}, {out['above_pfs_HR_hi']:.3f}]  "
              f"P={out['above_pfs_P']:.2e}")
        print(f"  BELOW n={out['below_n']:4d}  PFS HR={out['below_pfs_HR']:.3f} "
              f"[{out['below_pfs_HR_lo']:.3f}, {out['below_pfs_HR_hi']:.3f}]  "
              f"P={out['below_pfs_P']:.2e}")
        rev = (out['above_pfs_HR'] - 1) * (out['below_pfs_HR'] - 1) < 0
        print(f"  DIRECTION-REVERSED: {rev}")

    summary = pd.DataFrame(rows)
    summary.to_csv(VAL_DIR / "signature_metrics.tsv", sep="\t", index=False)
    print(f"\n→ saved {VAL_DIR / 'signature_metrics.tsv'}")

    # KM grid: K rows × 2 cols (above, below)
    fig, axes = plt.subplots(K, 2, figsize=(8, 2.4 * K), squeeze=False)
    for k, out in enumerate(sig_results):
        km_panel(axes[k, 0], df, out["above"],
                  time_col="pfs_t", event_col="pfs_e",
                  label=f"Sig {k} ABOVE τ̂0",
                  hr=out["above_pfs_HR"], p=out["above_pfs_P"], n=out["above_n"])
        km_panel(axes[k, 1], df, out["below"],
                  time_col="pfs_t", event_col="pfs_e",
                  label=f"Sig {k} BELOW τ̂0",
                  hr=out["below_pfs_HR"], p=out["below_pfs_P"], n=out["below_n"])
        gene_label = sig_results[k]["label"]
        fig.text(0.005, 1 - (k + 0.5) / K * 0.95,
                  gene_label[:60],
                  fontsize=8, va="center", ha="left", weight="bold",
                  rotation=90)

    fig.suptitle("PDAC: per-signature direction-reversal test (PFS Kaplan–Meier)",
                  fontsize=10)
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.97])
    fig.savefig(VAL_DIR / "fig_signature_KMs.pdf")
    fig.savefig(VAL_DIR / "fig_signature_KMs.png", dpi=150)
    plt.close(fig)
    print(f"→ saved {VAL_DIR / 'fig_signature_KMs.pdf'} + .png")


if __name__ == "__main__":
    main()
