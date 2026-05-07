"""PDAC signature heatmap (Panel A only, K=5).

Mirrors crc_signatures_figure.py's panel A:
  - Rows: top-loaded SNV genes (above) and CNA arms (below) across K signatures
  - Cell value: directional loading
        cell = v          for arms (label encodes sign(rho))
        cell = v*sign(rho) for SNVs (label fixed MUT)
  - Positive (red) = labeled direction loads positively on arm-1 (FOLFIRINOX) axis

Inputs:
  attribution_analysis/pdac_signatures/attribution_matrix.pkl
  attribution_analysis/pdac_signatures/pmd_K5_cv*.pkl
"""
from __future__ import annotations
import sys, pickle, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from core.arm_mapping import label_signature_feature


OUT = ROOT / "attribution_analysis" / "pdac_signatures"
FIG_DIR = OUT / "figure_panels"
DATA_DIR = OUT / "panel_data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _snv_rho_map(state, snv_features: list[str]) -> dict[str, float]:
    """Spearman rho(A_pg, mut_pg) for each SNV gene (1 = patient has any variant in g)."""
    from scipy.stats import spearmanr
    A = state["A"]
    features = state["features"]
    patients = state["patients"]
    bundle = pickle.load(open(ROOT / "msk_chord_latent_features_raw.pkl", "rb"))
    snv_meta = bundle["data_snv"].copy()
    snv_meta["pid"] = snv_meta["Tumor_Sample_Barcode"].str[:9]
    rho_map = {}
    for g in snv_features:
        if g not in features:
            rho_map[g] = 0.0
            continue
        j = features.index(g)
        A_g = A[:, j]
        mut = pd.Series(0, index=patients, dtype=float)
        for pid in snv_meta[snv_meta["Hugo_Symbol"] == g]["pid"].unique():
            if pid in mut.index:
                mut[pid] = 1.0
        rho, _ = spearmanr(A_g, mut.values)
        rho_map[g] = float(rho) if not np.isnan(rho) else 0.0
    return rho_map


def export_panel_a_data(state, pmd_res, *, top_n_per_sig: int = 8) -> dict:
    V = pmd_res["V"]
    D = pmd_res["D"]
    K = V.shape[1]
    features = state["features"]
    is_arm = state["is_arm"]
    arm_dir = state["arm_dir"]

    top_rows = set()
    for k in range(K):
        order = np.argsort(-np.abs(V[:, k]))
        for i in order[:top_n_per_sig]:
            if abs(V[i, k]) > 1e-10:
                top_rows.add(i)
    top_idx = sorted(top_rows)

    snv_idx = [i for i in top_idx if not is_arm[i]]
    arm_idx = [i for i in top_idx if is_arm[i]]
    snv_idx = sorted(snv_idx, key=lambda i: features[i])

    def arm_sort_key(i):
        f = features[i]
        chrom_part = f[:-1]
        arm_part = f[-1]
        if chrom_part in ("X", "Y"):
            chrom_num = {"X": 23, "Y": 24}[chrom_part]
        else:
            chrom_num = int(chrom_part)
        return (chrom_num, arm_part)
    arm_idx = sorted(arm_idx, key=arm_sort_key)
    row_order = snv_idx + arm_idx

    snv_features = [features[i] for i in snv_idx]
    snv_rho = _snv_rho_map(state, snv_features) if snv_features else {}

    rows_out = []
    for i in row_order:
        f = features[i]
        if is_arm[i]:
            rho = arm_dir.get(f, {}).get("rho", 0.0)
            ftype = "CNA arm"
            direction = "AMP" if rho > 0 else ("LOSS" if rho < 0 else "?")
            label = f"{f}_{direction}"
        else:
            rho = snv_rho.get(f, 0.0)
            ftype = "SNV"
            direction = "MUT"
            label = f"{f}_{direction}"
        row = {
            "feature": f, "type": ftype, "label": label,
            "biology_direction": direction, "rho": rho, "rho_sign": int(np.sign(rho)),
        }
        for k in range(K):
            row[f"sig{k}_raw"] = float(V[i, k])
        if ftype == "CNA arm":
            multiplier = 1.0
        else:
            multiplier = float(np.sign(rho)) if rho != 0 else 1.0
        for k in range(K):
            row[f"sig{k}_directional"] = float(V[i, k]) * multiplier
        rows_out.append(row)
    df_loadings = pd.DataFrame(rows_out)
    df_loadings.to_csv(DATA_DIR / "panelA_loadings.tsv", sep="\t", index=False)

    df_sigs = pd.DataFrame({
        "sig": list(range(K)),
        "D": [float(D[k]) for k in range(K)],
        "sum_v": [float(V[:, k].sum()) for k in range(K)],
        "n_nonzero_full": [int((np.abs(V[:, k]) > 1e-10).sum()) for k in range(K)],
        "n_nonzero_shown": [int((np.abs(V[row_order, k]) > 1e-10).sum())
                              for k in range(K)],
    })
    df_sigs.to_csv(DATA_DIR / "panelA_signatures.tsv", sep="\t", index=False)
    print(f"  → {DATA_DIR / 'panelA_loadings.tsv'}")
    print(f"  → {DATA_DIR / 'panelA_signatures.tsv'}")
    return {"loadings": df_loadings, "signatures": df_sigs,
              "snv_count": len(snv_idx), "arm_count": len(arm_idx),
              "row_order": row_order, "snv_rho": snv_rho}


def panel_a_signature_heatmap(state, pmd_res, *, figsize=(5.5, 7.0)):
    df_load = pd.read_csv(DATA_DIR / "panelA_loadings.tsv", sep="\t")
    K = pmd_res["V"].shape[1]
    D = pmd_res["D"]

    sig_cols = [f"sig{k}_directional" for k in range(K)]
    M = df_load[sig_cols].values
    row_labels = df_load["label"].tolist()
    snv_count = int((df_load["type"] == "SNV").sum())
    arm_count = int((df_load["type"] == "CNA arm").sum())

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.max(np.abs(M)) if np.max(np.abs(M)) > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm,
                     interpolation="nearest")

    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f"Sig {k}" for k in range(K)], fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]
            if abs(v) > 0.05:
                color = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center",
                          fontsize=5.5, color=color)

    if snv_count > 0 and arm_count > 0:
        ax.axhline(snv_count - 0.5, color="black", lw=0.7, alpha=0.6)
    n_total = snv_count + arm_count
    snv_y_axes = 1.0 - (snv_count - 1) / 2 / max(n_total - 1, 1)
    arm_y_axes = 1.0 - (snv_count + (arm_count - 1) / 2) / max(n_total - 1, 1)
    ax.text(-0.30, snv_y_axes, "SNV", rotation=90,
              ha="center", va="center", fontsize=8, fontweight="bold",
              transform=ax.transAxes)
    ax.text(-0.30, arm_y_axes, "CNA arm", rotation=90,
              ha="center", va="center", fontsize=8, fontweight="bold",
              transform=ax.transAxes)

    for k, d in enumerate(D):
        ax.text(k, -0.9, f"D={d:.0f}", ha="center", va="bottom",
                  fontsize=7, color="black", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.6)
    cbar.set_label(r"Directional loading  $v_{g,k}\cdot\mathrm{sign}(\rho_g)$",
                     fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("PDAC signature decomposition (FOLFIRINOX vs Gem/Abraxane)",
                  fontsize=10, pad=20)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "panelA_signature_heatmap.pdf",
                  bbox_inches="tight")
    fig.savefig(FIG_DIR / "panelA_signature_heatmap.png", dpi=200,
                  bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG_DIR / 'panelA_signature_heatmap.pdf'} + .png")


def export_panel_b_tp53_kras_data():
    """Copy the TP53/KRAS per-group metrics to panel_data/ for canonical lookup."""
    import shutil
    src = OUT / "tp53_kras_simple" / "tp53_kras_per_group_metrics.tsv"
    dst = DATA_DIR / "panelB_tp53_kras_metrics.tsv"
    shutil.copyfile(src, dst)
    print(f"  → {dst}")


def panel_b_tp53_kras_forest(figsize=(6.5, 3.0)):
    """Panel B: TP53/KRAS co-mutation rule (+ 17p LOSS refinement) — PFS arm HR forest.

    PDAC arms: FOLFIRINOX (arm=1, red, HR<1) vs Gem/Abraxane (arm=0, blue, HR>1).
    """
    import matplotlib.ticker as mtick
    metrics = pd.read_csv(DATA_DIR / "panelB_tp53_kras_metrics.tsv", sep="\t")

    desired_order = [
        "Both mut + 17p LOSS (TP53+/KRAS+/17p-)",
        "Both mut (TP53+ / KRAS+)",
        "TP53-only (TP53+ / KRAS-WT)",
        "KRAS-only (TP53-WT / KRAS+)",
        "Both WT (TP53-WT / KRAS-WT)",
    ]
    short_labels = {
        "Both mut + 17p LOSS (TP53+/KRAS+/17p-)": r"TP53$+$ / KRAS$+$ / 17p$-$",
        "Both mut (TP53+ / KRAS+)":               r"TP53$+$ / KRAS$+$",
        "TP53-only (TP53+ / KRAS-WT)":            r"TP53$+$ / KRAS$-$",
        "KRAS-only (TP53-WT / KRAS+)":            r"TP53$-$ / KRAS$+$",
        "Both WT (TP53-WT / KRAS-WT)":            r"TP53$-$ / KRAS$-$",
    }
    available = [g for g in desired_order if g in metrics["group"].values]
    metrics = metrics.set_index("group").loc[available].reset_index()
    metrics["display_label"] = metrics["group"].map(short_labels)

    fig, ax = plt.subplots(figsize=figsize)
    n_rows = len(metrics)
    y_pos = np.arange(n_rows)[::-1]

    for k in range(n_rows):
        if k % 2 == 0:
            ax.axhspan(y_pos[k] - 0.5, y_pos[k] + 0.5,
                         color="#f5f5f5", zorder=0)

    for i, row in metrics.iterrows():
        y = y_pos[i]
        hr, lo, hi = row["PFS_HR"], row["PFS_HR_lo"], row["PFS_HR_hi"]
        if not np.isfinite(hr):
            ax.text(1.0, y, "(insufficient n / arm imbalance)",
                      fontsize=8, va="center", ha="center", color="gray", style="italic")
            continue
        color = "#c0392b" if hr < 1.0 else "#2980b9"
        ax.errorbar(
            hr, y,
            xerr=[[max(0, hr - lo)], [max(0, hi - hr)]],
            fmt="D", color=color, ecolor=color, capsize=2.5,
            markersize=7, lw=1.4)

    ax.axvline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 2.5)
    xticks = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    ax.xaxis.set_major_locator(mtick.FixedLocator(xticks))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("PFS HR (FOLFIRINOX vs Gem/Abraxane), 95% CI", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics["display_label"].tolist(), fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_title("PDAC TP53 / KRAS co-mutation rule", fontsize=10, pad=8)

    x_text = 2.7
    ax.text(x_text, n_rows - 0.5, "n  /  P (Wald)",
              fontsize=8, va="bottom", ha="left", fontweight="bold")
    for i, row in metrics.iterrows():
        y = y_pos[i]
        if np.isfinite(row["PFS_HR"]):
            ax.text(x_text, y, f"n={int(row['n']):>4}   P={row['PFS_P']:.2g}",
                      fontsize=8, va="center", ha="left", family="monospace")
        else:
            ax.text(x_text, y, f"n={int(row['n']):>4}   P=NA",
                      fontsize=8, va="center", ha="left", family="monospace",
                      color="gray")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "panelB_tp53_kras_forest.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "panelB_tp53_kras_forest.png", dpi=200,
                  bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG_DIR / 'panelB_tp53_kras_forest.pdf'} + .png")


def main():
    with open(OUT / "attribution_matrix.pkl", "rb") as fh:
        state = pickle.load(fh)
    A = state["A"]
    print(f"loaded attribution matrix: A={A.shape}", flush=True)

    K_choice = sys.argv[1] if len(sys.argv) > 1 else "5"
    pmd_files = sorted(glob.glob(str(OUT / f"pmd_K{K_choice}_cv*.pkl")))
    if not pmd_files:
        raise FileNotFoundError(
            f"No PMD fit found for K={K_choice} in {OUT}. "
            f"Run pdac_signatures.py first.")
    pmd_path = pmd_files[0]
    with open(pmd_path, "rb") as fh:
        pmd = pickle.load(fh)
    res = pmd["res"]
    print(f"loaded PMD fit: {Path(pmd_path).name}, K={res['K']}, "
          f"expl_var={res['explained_var']:.3f}", flush=True)

    print("\n=== Rendering preview panels to figure_panels/ ===")
    export_panel_a_data(state, res)
    panel_a_signature_heatmap(state, res)

    if (OUT / "tp53_kras_simple" / "tp53_kras_per_group_metrics.tsv").exists():
        export_panel_b_tp53_kras_data()
        panel_b_tp53_kras_forest()
    else:
        print("[skip] panel B forest — run pdac_tp53_kras_simple_rule.py first")


if __name__ == "__main__":
    main()
