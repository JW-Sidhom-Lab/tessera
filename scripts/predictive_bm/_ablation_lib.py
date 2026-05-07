"""Three-way ablation library: LOCAL vs LOCAL+GLOBAL vs FULL.

Tests whether the foundation model's predictive signal lives in:
  LOCAL         = variant-intrinsic + local sequence attention only
                  (context-INDEPENDENT - same allele in different patients
                   gives same embedding)
  LOCAL+GLOBAL  = adds within-modality self-attention (patient-context)
                  (variant attends to other variants in same modality;
                   CNA attends to other CNA segments in same modality)
  FULL          = adds InfoNCE projection (256-d each modality)
                  (the projection was trained to align SNV-bag with
                   CNA-bag representations via contrastive learning, so
                   it carries IMPLICIT cross-modal SNV+CNA interaction)

Slice boundaries (verified empirically from variance decomposition):
  variant_features (1169-d):
    [0:49]      intrinsic (ref_alt + VAF)              context-INDEPENDENT
    [49:481]    3 local-attention blocks (5'/3' seq)   context-INDEPENDENT
    [481:913]   3 global self-attention blocks         WITHIN-MODALITY context
    [913:1169]  InfoNCE projection                     IMPLICIT cross-modal

  cna_features (688-d):
    [0:144]     intrinsic CNA embedding                context-INDEPENDENT
    [144:432]   2 self-attention blocks                WITHIN-MODALITY context
    [432:688]   InfoNCE projection                     IMPLICIT cross-modal
"""
from __future__ import annotations

import sys
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import RobustScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dr import fit_dr_learner, interaction_test, threshold_indifference
from scipy.stats import spearmanr

RAW_PKL = ROOT / "msk_chord_latent_features_raw.pkl"

# Slice configurations
ABLATIONS = {
    "LOCAL":         {"snv": (0, 481),  "cna": (0, 144), "label": "Local only\n(intrinsic + local sequence attention)"},
    "LOCAL+GLOBAL":  {"snv": (0, 913),  "cna": (0, 432), "label": "Local + Global\n(adds within-modality self-attention)"},
    "FULL":          {"snv": (0, 1169), "cna": (0, 688), "label": "Full (Local + Global + InfoNCE)\n(adds implicit SNV+CNA cross-modal alignment)"},
}


def banner(t):
    bar = "=" * 70
    print(f"\n{bar}\n{t}\n{bar}", flush=True)


def build_patient_features_sliced(snv_emb, cna_emb, snv_meta, cna_meta,
                                       snv_lo, snv_hi, cna_lo, cna_hi):
    """Per-patient mean+max with sliced embeddings."""
    snv_z = RobustScaler().fit_transform(snv_emb[:, snv_lo:snv_hi])
    cna_z = RobustScaler().fit_transform(cna_emb[:, cna_lo:cna_hi])
    snv_df = pd.DataFrame(snv_z, index=snv_meta["pid"].values)
    cna_df = pd.DataFrame(cna_z, index=cna_meta["pid"].values)
    snv_mean = snv_df.groupby(level=0).mean().add_prefix("snv_mean_")
    snv_max  = snv_df.groupby(level=0).max().add_prefix("snv_max_")
    cna_mean = cna_df.groupby(level=0).mean().add_prefix("cna_mean_")
    cna_max  = cna_df.groupby(level=0).max().add_prefix("cna_max_")
    snv_n = snv_df.groupby(level=0).size()
    cna_n = cna_df.groupby(level=0).size()
    pids = sorted(set(snv_mean.index) & set(cna_mean.index))
    out = pd.concat([snv_mean.loc[pids], snv_max.loc[pids],
                       cna_mean.loc[pids], cna_max.loc[pids]], axis=1)
    out["tmb_log1p"] = np.log1p(snv_n.reindex(pids).fillna(0).values)
    out["cna_log1p"] = np.log1p(cna_n.reindex(pids).fillna(0).values)
    return out


def fit_ablation(name, df, X, cfg):
    """Run DR-learner + sign-align. Returns dict with τ̂ and key metrics."""
    print(f"  Fitting {name} (X.shape={X.shape})...", flush=True)
    t0 = time.time()
    res = fit_dr_learner(
        df, X,
        pca_var=cfg["pca_var"],
        K_mu1=cfg["K_mu1"], K_mu0=cfg["K_mu0"], K_e=cfg["K_e"], K_tau=cfg["K_tau"],
        sp_mu1=cfg["sp_mu1"], sp_mu0=cfg["sp_mu0"], sp_e=cfg["sp_e"], sp_tau=cfg["sp_tau"],
        n_outer=10, n_inner=5, seed=42, horizon_months=36.0,
    )
    raw = res["tau_oof"]
    itx_raw = interaction_test(df, raw, time_col="pfs_t", event_col="pfs_e")
    sign = -1.0 if itx_raw["HR"] > 1.0 else +1.0
    tau = sign * raw
    itx_pfs = interaction_test(df, tau, time_col="pfs_t", event_col="pfs_e")
    itx_os  = interaction_test(df, tau, time_col="os_t",  event_col="os_e")
    tau0 = threshold_indifference(itx_pfs)
    print(f"    done in {time.time()-t0:.1f}s. PFS interaction P = {itx_pfs['P']:.3e}, "
          f"OS P = {itx_os['P']:.3e}, τ̂_0 = {tau0:.2f}", flush=True)
    return {
        "name": name, "n_dim": int(X.shape[1]),
        "tau": tau, "n_pca": res["n_pca_components"],
        "itx_pfs_HR": itx_pfs["HR"], "itx_pfs_P": itx_pfs["P"],
        "itx_os_HR": itx_os["HR"],  "itx_os_P": itx_os["P"],
        "tau0": float(tau0), "sign": float(sign),
    }


def per_arm_km(df, mask, time_col, event_col, arm1_label, arm0_label):
    """Within mask subgroup, per-arm KM curves. Return dict mapping label -> kmf."""
    sub = df[mask]
    if len(sub) == 0: return None
    out = {}
    for arm_v, label in [(1, arm1_label), (0, arm0_label)]:
        s = sub[sub["arm"] == arm_v]
        if len(s) < 1:
            out[label] = None; continue
        kmf = KaplanMeierFitter().fit(s[time_col], s[event_col], label=label)
        out[label] = kmf
    if (sub["arm"]==1).sum() >= 3 and (sub["arm"]==0).sum() >= 3:
        try:
            cph = CoxPHFitter().fit(sub[[time_col,event_col,"arm"]].rename(columns={"arm":"X"}),
                                          duration_col=time_col, event_col=event_col)
            hr = float(np.exp(cph.params_["X"]))
            p  = float(cph.summary.loc["X","p"])
        except Exception:
            hr, p = None, None
    else:
        hr, p = None, None
    return out, hr, p, len(sub), int((sub["arm"]==1).sum()), int((sub["arm"]==0).sum())


def make_figure(cohort_name, df, ablations, full_tau_oof, out_path,
                  arm1_label="FOLFOX", arm0_label="FOLFIRI"):
    """3 cols (one per ablation) × 3 rows: τ̂ histogram, above-τ̂_0 KM, below-τ̂_0 KM."""
    mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                            "axes.labelsize": 9, "xtick.labelsize": 8,
                            "ytick.labelsize": 8, "figure.titlesize": 11,
                            "savefig.dpi": 200, "figure.dpi": 130})

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    arm1_color = "#3a78b8"
    arm0_color = "#d34d3a"

    for col, (key, info) in enumerate(ablations.items()):
        tau = info["tau"]
        tau0 = info["tau0"]
        rho_full = float(spearmanr(tau, full_tau_oof)[0])
        n_above = int((tau > tau0).sum())
        n_below = int((tau <= tau0).sum())

        # Row 0: τ̂ histogram
        axh = axes[0, col]
        axh.hist(tau, bins=40, color="#7799c8", edgecolor="white", alpha=0.85)
        axh.axvline(tau0, ls="--", color="black", lw=1, label=f"τ̂_0 = {tau0:.1f}")
        axh.set_xlabel("τ̂_OOF")
        axh.set_ylabel("# patients")
        title = (f"{info['label']}\n"
                  f"n_dim = {info['n_dim']}\n"
                  f"PFS interaction P = {info['itx_pfs_P']:.2e}\n"
                  f"Spearman vs FULL = {rho_full:+.3f}\n"
                  f"above τ̂_0: {n_above}/{len(tau)} ({100*n_above/len(tau):.0f}%); "
                  f"below: {n_below}")
        axh.set_title(title, fontsize=8.5)
        axh.legend(loc="upper right", frameon=False, fontsize=7)

        # Row 1: KM above τ̂_0 (arm=1-favored predicted region)
        axa = axes[1, col]
        if n_above >= 6:
            mask_above = tau > tau0
            kms_a, hr_a, p_a, n_a, n1_a, n0_a = per_arm_km(df, mask_above, "pfs_t", "pfs_e",
                                                                       arm1_label, arm0_label)
            for label, kmf in kms_a.items():
                if kmf is None: continue
                color = arm1_color if label == arm1_label else arm0_color
                kmf.plot(ax=axa, ci_show=False, color=color, lw=2)
            axa.set_xlim(0, 36); axa.set_ylim(0, 1)
            axa.set_xlabel("PFS (months)"); axa.set_ylabel("Surv prob")
            arm_hr_str = (f"HR({arm1_label} vs {arm0_label}) = {hr_a:.2f}, P = {p_a:.2e}"
                            if hr_a is not None else "arm-HR n/a")
            axa.set_title(f"Above τ̂_0 (n={n_a};\n"
                            f"{arm1_label}={n1_a}, {arm0_label}={n0_a})\n{arm_hr_str}",
                            fontsize=8.5)
            axa.legend(loc="upper right", frameon=False, fontsize=7)
        else:
            axa.text(0.5, 0.5, f"n_above = {n_above}\n(too few to plot)",
                       ha="center", va="center", transform=axa.transAxes, fontsize=11)
            axa.set_xticks([]); axa.set_yticks([])

        # Row 2: KM below τ̂_0 (arm=0-favored predicted region)
        axb = axes[2, col]
        if n_below >= 6:
            mask_below = tau <= tau0
            kms_b, hr_b, p_b, n_b, n1_b, n0_b = per_arm_km(df, mask_below, "pfs_t", "pfs_e",
                                                                       arm1_label, arm0_label)
            for label, kmf in kms_b.items():
                if kmf is None: continue
                color = arm1_color if label == arm1_label else arm0_color
                kmf.plot(ax=axb, ci_show=False, color=color, lw=2)
            axb.set_xlim(0, 36); axb.set_ylim(0, 1)
            axb.set_xlabel("PFS (months)"); axb.set_ylabel("Surv prob")
            arm_hr_str = (f"HR({arm1_label} vs {arm0_label}) = {hr_b:.2f}, P = {p_b:.2e}"
                            if hr_b is not None else "arm-HR n/a")
            axb.set_title(f"Below τ̂_0 (n={n_b};\n"
                            f"{arm1_label}={n1_b}, {arm0_label}={n0_b})\n{arm_hr_str}",
                            fontsize=8.5)
            axb.legend(loc="upper right", frameon=False, fontsize=7)
        else:
            axb.text(0.5, 0.5, f"n_below = {n_below}\n(too few to plot)",
                       ha="center", va="center", transform=axb.transAxes, fontsize=11)
            axb.set_xticks([]); axb.set_yticks([])

    fig.suptitle(f"{cohort_name}: feature-ablation comparison ({arm1_label} vs {arm0_label})",
                   fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved figure to {out_path}", flush=True)


def run_cohort(cohort_name, gt_path, builder, cfg, arm1_label, arm0_label, out_dir):
    """Top-level entry: load data, run 3 ablations, save predictions, make figure."""
    print(f"\nLoading raw...", flush=True)
    with open(RAW_PKL, "rb") as f: raw = pickle.load(f)
    snv_emb = raw["variant_features"]
    cna_emb = raw["cna_features"]
    snv_meta = raw["data_snv"][["Tumor_Sample_Barcode"]].copy()
    cna_meta = raw["data_cna"][["Tumor_Sample_Barcode"]].copy()
    snv_meta["pid"] = snv_meta["Tumor_Sample_Barcode"].str[:9]
    cna_meta["pid"] = cna_meta["Tumor_Sample_Barcode"].str[:9]
    print(f"  variant_features: {snv_emb.shape}", flush=True)
    print(f"  cna_features: {cna_emb.shape}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Run all 3 ablations
    ablations = {}
    for name, slices in ABLATIONS.items():
        banner(f"{cohort_name} - {name}")
        feat = build_patient_features_sliced(snv_emb, cna_emb, snv_meta, cna_meta,
                                                    *slices["snv"], *slices["cna"])
        df, X = builder(gt_path, feat, horizon_months=36.0)
        fit_res = fit_ablation(name, df, X, cfg)
        fit_res["label"] = slices["label"]
        ablations[name] = fit_res

        # Save predictions
        pred_df = df[["pid","arm","pfs_t","pfs_e","os_t","os_e"]].copy()
        pred_df["tau"] = fit_res["tau"]
        pred_csv = out_dir / f"{cohort_name.lower()}_predictions_{name.lower().replace('+','_')}.csv"
        pred_df.to_csv(pred_csv, index=False)
        print(f"  saved {pred_csv}", flush=True)

    # Make summary table
    rows = [{"ablation": name,
                "n_dim": a["n_dim"],
                "n_pca_components": a["n_pca"],
                "PFS_interaction_HR": a["itx_pfs_HR"],
                "PFS_interaction_P":  a["itx_pfs_P"],
                "OS_interaction_HR":  a["itx_os_HR"],
                "OS_interaction_P":   a["itx_os_P"],
                "tau0": a["tau0"],
                "n_above_tau0": int((a["tau"] > a["tau0"]).sum()),
                "n_below_tau0": int((a["tau"] <= a["tau0"]).sum()),
                "spearman_vs_FULL": float(spearmanr(a["tau"], ablations["FULL"]["tau"])[0]),
            } for name, a in ablations.items()]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / f"{cohort_name.lower()}_ablation_summary.tsv",
                         sep="\t", index=False)
    print(f"\n{cohort_name} summary:", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    # Figure
    figure_path = out_dir / f"{cohort_name.lower()}_ablation_figure.png"
    make_figure(cohort_name, df, ablations, ablations["FULL"]["tau"], figure_path,
                  arm1_label=arm1_label, arm0_label=arm0_label)
