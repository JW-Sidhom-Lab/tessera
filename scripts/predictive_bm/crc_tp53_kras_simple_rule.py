"""CRC TP53 / KRAS / 17p genotype-rule subgroups.

Stratifies CRC patients into five exclusive subgroups along the
manuscript's two-way rule axis (TP53+/KRAS+/17p-LOSS, TP53+/KRAS+,
TP53+/KRAS-, TP53-/KRAS+, TP53-/KRAS-) and fits per-subgroup
arm-vs-PFS Cox + KM curves. Produces the data backing Fig 6 k-m.
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from core.dr import per_group_arm_hr
from core.arm_mapping import build_arm_coords


OUT = ROOT / "attribution_analysis" / "crc_signatures"
VAL = OUT / "tp53_kras_simple"
VAL.mkdir(parents=True, exist_ok=True)


def safe_arm_hr(df, mask, time_col, event_col):
    sub = df[mask]
    nan = {"n": int(len(sub)), "HR": float("nan"), "HR_lo": float("nan"),
            "HR_hi": float("nan"), "P": float("nan")}
    if len(sub) < 10: return nan
    arm_counts = sub["arm"].value_counts()
    if len(arm_counts) < 2 or (arm_counts < 5).any(): return nan
    if sub[event_col].sum() < 3: return nan
    try:
        return per_group_arm_hr(df, mask, time_col=time_col, event_col=event_col)
    except Exception:
        return nan


def km_panel(ax, df, mask, time_col, event_col, label, hr_info,
              arm_colors=("#d62728", "#1f77b4"),
              arm_labels=("FOLFOX", "FOLFIRI"), xlim=24.0):
    sub = df[mask]
    for arm_v, color, alabel in zip([1, 0], arm_colors, arm_labels):
        s = sub[sub["arm"] == arm_v]
        if len(s) < 2: continue
        kmf = KaplanMeierFitter()
        kmf.fit(s[time_col].values, event_observed=s[event_col].values, label=alabel)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, lw=1.5)
    ax.set_xlim(0, xlim); ax.set_ylim(0, 1.02)
    title = f"{label}\nn={hr_info['n']}  HR={hr_info.get('PFS_HR', float('nan')):.2f}  P={hr_info.get('PFS_P', float('nan')):.2e}"
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("PFS (months)", fontsize=8)
    ax.set_ylabel("Survival", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, frameon=False)


def main():
    state = pickle.load(open(OUT / "attribution_matrix.pkl", "rb"))
    df = state["df"]
    patients = state["patients"]

    bundle = pickle.load(open(ROOT / "msk_chord_latent_features_raw.pkl", "rb"))
    snv_meta = bundle["data_snv"].copy()
    snv_meta["pid"] = snv_meta["Tumor_Sample_Barcode"].str[:9]
    mut = pd.DataFrame(0, index=patients, columns=["TP53", "KRAS"], dtype=float)
    for _, r in snv_meta[snv_meta["Hugo_Symbol"].isin(["TP53", "KRAS"])].iterrows():
        if r["pid"] in mut.index:
            mut.loc[r["pid"], r["Hugo_Symbol"]] = 1.0

    # Compute per-patient 17p log2 (length-weighted mean across 17p-overlapping segments)
    cna_meta = bundle["data_cna"].copy()
    cna_meta["pid"] = cna_meta["Tumor_Sample_Barcode"].str[:9]
    arms = build_arm_coords()
    arm_17p = arms[arms["arm"] == "17p"].iloc[0]
    rows_17p = []
    for _, seg in cna_meta[cna_meta["Chromosome"].astype(str) == "17"].iterrows():
        ov = max(0, min(int(seg["End"]), int(arm_17p["end"]))
                    - max(int(seg["Start"]), int(arm_17p["start"])))
        if ov > 0:
            rows_17p.append((seg["pid"], seg["Segment_Mean"], ov))
    raw17 = pd.DataFrame(rows_17p, columns=["pid", "log2", "ov"])
    raw17["w_log2"] = raw17["log2"] * raw17["ov"]
    per17 = (raw17.groupby("pid")
                  .agg(s=("w_log2", "sum"), w=("ov", "sum")).reset_index())
    per17["log2_17p"] = per17["s"] / per17["w"]
    log2_17p = per17.set_index("pid")["log2_17p"].reindex(patients).fillna(0.0)
    has_17p_loss = (log2_17p < -0.2).astype(int)

    pid_to_mut = mut.copy()
    pid_to_mut["has_17p_loss"] = has_17p_loss
    df_with_mut = df.set_index("pid").join(pid_to_mut, how="left").reset_index()
    df_with_mut["TP53"] = df_with_mut["TP53"].fillna(0).astype(int)
    df_with_mut["KRAS"] = df_with_mut["KRAS"].fillna(0).astype(int)
    df_with_mut["has_17p_loss"] = df_with_mut["has_17p_loss"].fillna(0).astype(int)

    # 4 base TP53/KRAS groups + 1 refined Both-mut subgroup with 17p LOSS
    groups = {
        "Both mut (TP53+ / KRAS+)":              (df_with_mut["TP53"] == 1) & (df_with_mut["KRAS"] == 1),
        "Both mut + 17p LOSS (TP53+/KRAS+/17p-)":(df_with_mut["TP53"] == 1) & (df_with_mut["KRAS"] == 1) & (df_with_mut["has_17p_loss"] == 1),
        "TP53-only (TP53+ / KRAS-WT)":           (df_with_mut["TP53"] == 1) & (df_with_mut["KRAS"] == 0),
        "KRAS-only (TP53-WT / KRAS+)":           (df_with_mut["TP53"] == 0) & (df_with_mut["KRAS"] == 1),
        "Both WT (TP53-WT / KRAS-WT)":           (df_with_mut["TP53"] == 0) & (df_with_mut["KRAS"] == 0),
    }

    print("=" * 70)
    print("Simple TP53/KRAS co-mutation rule: per-group PFS arm HR")
    print("=" * 70)
    rows = []
    for label, mask in groups.items():
        hr_pfs = safe_arm_hr(df_with_mut, mask.values, "pfs_t", "pfs_e")
        hr_os  = safe_arm_hr(df_with_mut, mask.values, "os_t",  "os_e")
        rows.append({
            "group": label, "n": hr_pfs["n"],
            "PFS_HR": hr_pfs["HR"], "PFS_HR_lo": hr_pfs["HR_lo"],
            "PFS_HR_hi": hr_pfs["HR_hi"], "PFS_P": hr_pfs["P"],
            "OS_HR": hr_os["HR"], "OS_P": hr_os["P"],
        })
        favor = "FOLFOX" if hr_pfs["HR"] < 1 else ("FOLFIRI" if hr_pfs["HR"] > 1 else "tie")
        print(f"\n{label}  n={hr_pfs['n']:4d}")
        print(f"  PFS  HR(FOLFOX vs FOLFIRI) = {hr_pfs['HR']:.3f}  "
              f"[{hr_pfs['HR_lo']:.3f}, {hr_pfs['HR_hi']:.3f}]  P={hr_pfs['P']:.2e}  "
              f"-> favors {favor}")
        print(f"  OS   HR(FOLFOX vs FOLFIRI) = {hr_os['HR']:.3f}  P={hr_os['P']:.2e}")

    summary = pd.DataFrame(rows)
    summary.to_csv(VAL / "tp53_kras_per_group_metrics.tsv", sep="\t", index=False)
    print(f"\n→ {VAL / 'tp53_kras_per_group_metrics.tsv'}")

    # KM grid: handle however many groups we have (currently 5)
    n_groups = len(groups)
    n_cols = 2
    n_rows = int(np.ceil(n_groups / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 3.0 * n_rows),
                                squeeze=False)
    axflat = axes.flatten()
    for i, (label, mask) in enumerate(groups.items()):
        hr_pfs = rows[i]
        km_panel(axflat[i], df_with_mut, mask.values,
                  "pfs_t", "pfs_e", label, hr_pfs)
    for j in range(n_groups, len(axflat)):
        axflat[j].axis("off")
    fig.suptitle("CRC: PFS by treatment arm, stratified by TP53/KRAS co-mutation status",
                  fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(VAL / "tp53_kras_KMs.pdf")
    fig.savefig(VAL / "tp53_kras_KMs.png", dpi=150)
    plt.close(fig)
    print(f"→ {VAL / 'tp53_kras_KMs.pdf'} + .png")


if __name__ == "__main__":
    main()
