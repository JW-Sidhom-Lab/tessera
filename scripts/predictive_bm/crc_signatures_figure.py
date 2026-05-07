"""Build Figure 6 bottom panels (signature decomposition).

Panel A: V-matrix heatmap (features × signatures) showing the top-loaded
        features for each of the 5 PMD signatures, with feature-type
        and biology-direction annotations.
Panel B: Per-signature × per-stratum forest plot of PFS arm HRs.
Panel C: Sig 4 spotlight (loadings + co-mutation enrichment + KMs).
Panel D: TP53/KRAS 4-group simple-rule forest.

This script renders each panel independently as a PDF/PNG. The manuscript
build pipeline then composes them into the bottom-of-Fig-6 layout.
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
from matplotlib.colors import TwoSlopeNorm

from core.arm_mapping import label_signature_feature


OUT = ROOT / "attribution_analysis" / "crc_signatures"
FIG_DIR = OUT / "figure_panels"
DATA_DIR = OUT / "panel_data"     # csv/tsv files for manuscript/build/ to read
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Panel A — V matrix heatmap
# ============================================================================

def _snv_rho_map(state, snv_features: list[str]) -> dict[str, float]:
    """Compute Spearman rho(A_pg, mut_pg) for each SNV gene in snv_features.
    Mut indicator is 1 if patient has any variant in gene g, else 0.
    """
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
    """Compute Panel A's data and write to TSV. Returns a dict that the
    rendering function uses, so we don't recompute.

    Each feature's biology direction is determined by the empirical correlation
    between its attribution column and its raw biological observable (log2 ratio
    for arms; mutation indicator for SNVs). We then display directional loadings
    v_g,k * sign(rho_g) so that positive cells uniformly mean "feature in its
    labeled direction loads positively on the signature axis."

    Files written:
      panel_data/panelA_loadings.tsv     (feature x signature loadings, both raw and directional)
      panel_data/panelA_signatures.tsv   (per-signature D, n_nonzero, etc.)
    """
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

    # Compute SNV rho map for top SNV features
    snv_features = [features[i] for i in snv_idx]
    snv_rho = _snv_rho_map(state, snv_features) if snv_features else {}

    # Build TSV with both raw and directional loadings
    rows_out = []
    for i in row_order:
        f = features[i]
        if is_arm[i]:
            rho = arm_dir.get(f, {}).get("rho", 0.0)
            ftype = "CNA arm"
            direction = "AMP" if rho > 0 else ("LOSS" if rho < 0 else "?")
            label = f"{f}_{direction}"
        else:
            # SNVs: always label as MUT (WT direction is implicit / has zero attribution).
            # Cell value v * sign(rho) makes the loading direction-aware regardless.
            rho = snv_rho.get(f, 0.0)
            ftype = "SNV"
            direction = "MUT"
            label = f"{f}_{direction}"
        row = {
            "feature": f, "type": ftype, "label": label,
            "biology_direction": direction, "rho": rho, "rho_sign": int(np.sign(rho)),
        }
        # Raw v (PMD output)
        for k in range(K):
            row[f"sig{k}_raw"] = float(V[i, k])
        # Directional cell value, full PMD-loading magnitude preserved:
        #   - Arms (label by sign(rho)):  cell = v        (label encodes direction)
        #   - SNVs (label fixed MUT):     cell = v * sign(rho)  (sign correction)
        # Positive cell = labeled direction loads positively on the FOLFOX axis.
        if ftype == "CNA arm":
            multiplier = 1.0
        else:  # SNV / MUT
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


def panel_a_signature_heatmap(
    state, pmd_res, *,
    top_n_per_sig: int = 8,
    figsize=(5.5, 7.0),
):
    """Heatmap of DIRECTIONAL loadings v * sign(rho) across 5 signatures.

    Each row label is "feature_DIRECTION" (AMP/LOSS for arms, MUT/WT for SNVs)
    where DIRECTION is determined by the empirical correlation between the
    feature's attribution column and its raw biological observable (log2
    ratio for arms; mutation indicator for SNVs).

    Cell value = v_g,k * sign(rho_g). Positive (red) cells uniformly mean
    "feature in its labeled direction loads positively on the signature axis."
    No mental multiplication required.
    """
    # Read directly from the panel data we just wrote
    df_load = pd.read_csv(DATA_DIR / "panelA_loadings.tsv", sep="\t")
    K = pmd_res["V"].shape[1]
    D = pmd_res["D"]

    sig_cols = [f"sig{k}_directional" for k in range(K)]
    M = df_load[sig_cols].values
    row_labels = df_load["label"].tolist()
    snv_count = int((df_load["type"] == "SNV").sum())
    arm_count = int((df_load["type"] == "CNA arm").sum())

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.max(np.abs(M)) if np.max(np.abs(M)) > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm,
                     interpolation="nearest")

    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f"Sig {k}" for k in range(K)], fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Annotate text values where |v| is meaningful
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]
            if abs(v) > 0.05:
                color = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center",
                          fontsize=5.5, color=color)

    # SNV/arm separator
    if snv_count > 0 and arm_count > 0:
        ax.axhline(snv_count - 0.5, color="black", lw=0.7, alpha=0.6)
    # Group labels in axes fraction coords, well to the left of y-tick labels
    n_total = snv_count + arm_count
    snv_y_axes = 1.0 - (snv_count - 1) / 2 / max(n_total - 1, 1)
    arm_y_axes = 1.0 - (snv_count + (arm_count - 1) / 2) / max(n_total - 1, 1)
    ax.text(-0.30, snv_y_axes, "SNV", rotation=90,
              ha="center", va="center", fontsize=8, fontweight="bold",
              transform=ax.transAxes)
    ax.text(-0.30, arm_y_axes, "CNA arm", rotation=90,
              ha="center", va="center", fontsize=8, fontweight="bold",
              transform=ax.transAxes)

    # D values across the top
    for k, d in enumerate(D):
        ax.text(k, -0.9, f"D={d:.0f}", ha="center", va="bottom",
                  fontsize=7, color="black", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.6)
    cbar.set_label(r"Directional loading  $v_{g,k}\cdot\mathrm{sign}(\rho_g)$",
                     fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Signature decomposition of attribution matrix",
                  fontsize=10, pad=20)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "panelA_signature_heatmap.pdf",
                  bbox_inches="tight")
    fig.savefig(FIG_DIR / "panelA_signature_heatmap.png", dpi=200,
                  bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG_DIR / 'panelA_signature_heatmap.pdf'} + .png")


def panel_b_per_signature_forest(figsize=(6.5, 4.0)):
    """Per-signature x per-stratum PFS arm-HR forest plot.

    Standard forest layout: one row per (signature, stratum), HR + 95% CI
    marker, n and P annotated to the right. Subtle horizontal band per
    signature for visual grouping. Reference line at HR=1.
    """
    metrics = pd.read_csv(
        OUT / "signature_validation" / "signature_metrics.tsv", sep="\t")
    K = len(metrics)

    above_color, below_color = "#c0392b", "#2980b9"  # FOLFOX (red) / FOLFIRI (blue)

    # Build row list: [(sig, "ABOVE"|"BELOW", row_data), ...]
    plot_rows = []
    for _, row in metrics.iterrows():
        plot_rows.append((int(row["sig"]), "ABOVE", row))
        plot_rows.append((int(row["sig"]), "BELOW", row))

    n_rows = len(plot_rows)
    y_pos = np.arange(n_rows)[::-1]   # row 0 at top

    fig, ax = plt.subplots(figsize=figsize)

    # Background bands (alternate per signature)
    for k in range(K):
        y_top = n_rows - 1 - 2 * k + 0.5
        y_bot = n_rows - 1 - 2 * k - 1.5
        if k % 2 == 0:
            ax.axhspan(y_bot, y_top, color="#f5f5f5", zorder=0)

    for i, (sig, stratum, row) in enumerate(plot_rows):
        y = y_pos[i]
        if stratum == "ABOVE":
            hr, lo, hi, n, p = (
                row["above_PFS_HR"], row["above_PFS_HR_lo"],
                row["above_PFS_HR_hi"], int(row["above_n"]), row["above_PFS_P"])
            color = above_color
        else:
            hr, lo, hi, n, p = (
                row["below_PFS_HR"], row["below_PFS_HR_lo"],
                row["below_PFS_HR_hi"], int(row["below_n"]), row["below_PFS_P"])
            color = below_color

        if np.isnan(hr) or np.isnan(lo) or np.isnan(hi):
            ax.text(1.0, y, f"degenerate (n={n})",
                     fontsize=7, va="center", ha="center",
                     color="gray", style="italic")
            continue

        ax.errorbar(hr, y,
                      xerr=[[max(0, hr - lo)], [max(0, hi - hr)]],
                      fmt="s" if stratum == "ABOVE" else "o",
                      color=color, ecolor=color, capsize=2.5,
                      markersize=6, lw=1.2)

    import matplotlib.ticker as mtick
    ax.axvline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 4.0)
    xticks = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    ax.xaxis.set_major_locator(mtick.FixedLocator(xticks))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("PFS HR (FOLFOX vs FOLFIRI), 95% CI", fontsize=9)

    # Y-axis labels: signature index + stratum
    yticklabels = []
    for sig, stratum, _ in plot_rows:
        yticklabels.append(f"Sig {sig}  {stratum}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.tick_params(axis="y", length=0)

    # Right-side annotation column: n and P
    x_text = 4.4   # outside x-limit, in data coords
    for i, (sig, stratum, row) in enumerate(plot_rows):
        y = y_pos[i]
        if stratum == "ABOVE":
            n, p = int(row["above_n"]), row["above_PFS_P"]
        else:
            n, p = int(row["below_n"]), row["below_PFS_P"]
        if not np.isnan(p):
            ax.text(x_text, y, f"n={n:>4}   P={p:.2g}",
                     fontsize=8, va="center", ha="left",
                     family="monospace")
        else:
            ax.text(x_text, y, f"n={n:>4}",
                     fontsize=8, va="center", ha="left",
                     family="monospace", color="gray")

    # Header for the right column
    ax.text(x_text, n_rows - 0.4, "n  /  P (Wald)",
              fontsize=8, va="bottom", ha="left", fontweight="bold")

    ax.set_title("Per-signature direction-reversal test", fontsize=10, pad=10)
    ax.set_ylim(-0.7, n_rows - 0.3)

    # Legend (compact, outside)
    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=above_color,
                    markersize=7, label=r"ABOVE $\hat\tau_0$ (predicted FOLFOX)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=below_color,
                    markersize=7, label=r"BELOW $\hat\tau_0$ (predicted FOLFIRI)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7,
                loc="upper left", bbox_to_anchor=(0, -0.18),
                frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "panelB_per_signature_forest.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "panelB_per_signature_forest.png", dpi=200,
                  bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG_DIR / 'panelB_per_signature_forest.pdf'} + .png")


def export_panel_b_data():
    """Panel B uses signature_validation/signature_metrics.tsv directly.
    Copy/symlink it to panel_data/ for a single canonical location.
    """
    import shutil
    src = OUT / "signature_validation" / "signature_metrics.tsv"
    dst = DATA_DIR / "panelB_signature_metrics.tsv"
    shutil.copyfile(src, dst)
    print(f"  → {dst}")


def export_panel_b_tp53_kras_data():
    """Panel B (TP53/KRAS forest) uses tp53_kras_simple/tp53_kras_per_group_metrics.tsv."""
    import shutil
    src = OUT / "tp53_kras_simple" / "tp53_kras_per_group_metrics.tsv"
    dst = DATA_DIR / "panelB_tp53_kras_metrics.tsv"
    shutil.copyfile(src, dst)
    print(f"  → {dst}")


def _build_genotype_df():
    """Build a DataFrame of CRC patients with TP53, KRAS, and has_17p_loss columns."""
    state = pickle.load(open(OUT / "attribution_matrix.pkl", "rb"))
    df = state["df"]
    patients = state["patients"]

    bundle = pickle.load(open(ROOT / "msk_chord_latent_features_raw.pkl", "rb"))
    snv_meta = bundle["data_snv"].copy()
    snv_meta["pid"] = snv_meta["Tumor_Sample_Barcode"].str[:9]
    cna_meta = bundle["data_cna"].copy()
    cna_meta["pid"] = cna_meta["Tumor_Sample_Barcode"].str[:9]

    mut = pd.DataFrame(0, index=patients, columns=["TP53", "KRAS"], dtype=float)
    for _, r in snv_meta[snv_meta["Hugo_Symbol"].isin(["TP53", "KRAS"])].iterrows():
        if r["pid"] in mut.index:
            mut.loc[r["pid"], r["Hugo_Symbol"]] = 1.0

    from core.arm_mapping import build_arm_coords
    arm_17p = build_arm_coords()
    arm_17p = arm_17p[arm_17p["arm"] == "17p"].iloc[0]
    rows_17p = []
    for _, seg in cna_meta[cna_meta["Chromosome"].astype(str) == "17"].iterrows():
        ov = max(0, min(int(seg["End"]), int(arm_17p["end"]))
                    - max(int(seg["Start"]), int(arm_17p["start"])))
        if ov > 0:
            rows_17p.append((seg["pid"], seg["Segment_Mean"], ov))
    raw17 = pd.DataFrame(rows_17p, columns=["pid", "log2", "ov"])
    raw17["w_log2"] = raw17["log2"] * raw17["ov"]
    per17 = (raw17.groupby("pid").agg(s=("w_log2", "sum"), w=("ov", "sum")).reset_index())
    per17["log2_17p"] = per17["s"] / per17["w"]
    log2_17p = per17.set_index("pid")["log2_17p"].reindex(patients).fillna(0.0)
    has_17p_loss = (log2_17p < -0.2).astype(int)

    df_with = df.set_index("pid").join(mut, how="left").join(
        pd.Series(has_17p_loss, name="has_17p_loss"), how="left").reset_index()
    df_with["TP53"] = df_with["TP53"].fillna(0).astype(int)
    df_with["KRAS"] = df_with["KRAS"].fillna(0).astype(int)
    df_with["has_17p_loss"] = df_with["has_17p_loss"].fillna(0).astype(int)
    return df_with


def _km_subgroup_panel(df_with, mask, *, group_canonical_name: str,
                          title: str, panel_letter: str,
                          output_basename: str, data_filename: str,
                          figsize=(4.5, 3.5)):
    """Render a within-subgroup KM split by treatment arm. Reads HR/CI/P from
    the saved Panel B metrics TSV so the panels are statistically consistent.
    """
    from lifelines import KaplanMeierFitter
    sub = df_with[mask]
    n_total = len(sub)
    n_fox = int((sub["arm"] == 1).sum())
    n_fri = int((sub["arm"] == 0).sum())

    metrics = pd.read_csv(DATA_DIR / "panelB_tp53_kras_metrics.tsv", sep="\t")
    row = metrics[metrics["group"] == group_canonical_name].iloc[0]
    hr = float(row["PFS_HR"])
    hr_lo = float(row["PFS_HR_lo"])
    hr_hi = float(row["PFS_HR_hi"])
    p = float(row["PFS_P"])

    # Save panel data
    pd.DataFrame({
        "pid":   sub["pid"].values,
        "arm":   sub["arm"].values,
        "pfs_t": sub["pfs_t"].values,
        "pfs_e": sub["pfs_e"].values,
    }).to_csv(DATA_DIR / data_filename, sep="\t", index=False)
    print(f"  → {DATA_DIR / data_filename}")

    fig, ax = plt.subplots(figsize=figsize)
    arm_colors = ("#c0392b", "#2980b9")  # FOLFOX, FOLFIRI
    arm_labels = ("FOLFOX", "FOLFIRI")
    for arm_v, color, label in zip([1, 0], arm_colors, arm_labels):
        s = sub[sub["arm"] == arm_v]
        if len(s) < 2:
            continue
        kmf = KaplanMeierFitter().fit(
            s["pfs_t"].values, event_observed=s["pfs_e"].values, label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, lw=1.6,
                                       ci_alpha=0.15)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PFS (months)", fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.set_title(title, fontsize=10, pad=8)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)

    annot = (f"n = {n_total}  (FOLFOX = {n_fox}, FOLFIRI = {n_fri})\n"
              f"HR = {hr:.2f}  [{hr_lo:.2f}, {hr_hi:.2f}]   P = {p:.2g}")
    ax.text(0.02, 0.04, annot, transform=ax.transAxes,
              fontsize=8, va="bottom", ha="left",
              bbox={"facecolor": "white", "edgecolor": "lightgray",
                      "boxstyle": "round,pad=0.3"})

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{output_basename}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{output_basename}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG_DIR / output_basename}.pdf + .png")


def panel_c_triple_mutant_km(figsize=(4.5, 3.5)):
    """Panel C: KM for TP53+/KRAS+/17p- triple-mutant subgroup (FOLFOX-favored).
    """
    df_with = _build_genotype_df()
    mask = (df_with["TP53"] == 1) & (df_with["KRAS"] == 1) & (df_with["has_17p_loss"] == 1)
    _km_subgroup_panel(
        df_with, mask,
        group_canonical_name="Both mut + 17p LOSS (TP53+/KRAS+/17p-)",
        title=r"TP53$+$ / KRAS$+$ / 17p$-$ subgroup",
        panel_letter="C",
        output_basename="panelC_triple_mutant_KM",
        data_filename="panelC_triple_mutant_KM.tsv",
        figsize=figsize,
    )


def panel_d_double_wt_km(figsize=(4.5, 3.5)):
    """Panel D: KM for TP53-/KRAS- double-WT subgroup (FOLFIRI-favored, mirror of C).
    """
    df_with = _build_genotype_df()
    mask = (df_with["TP53"] == 0) & (df_with["KRAS"] == 0)
    _km_subgroup_panel(
        df_with, mask,
        group_canonical_name="Both WT (TP53-WT / KRAS-WT)",
        title=r"TP53$-$ / KRAS$-$ subgroup",
        panel_letter="D",
        output_basename="panelD_double_wt_KM",
        data_filename="panelD_double_wt_KM.tsv",
        figsize=figsize,
    )


def panel_b_tp53_kras_forest(figsize=(6.5, 3.0)):
    """Panel B: TP53/KRAS co-mutation rule (+ 17p LOSS refinement) — PFS arm HR forest.
    """
    import matplotlib.ticker as mtick
    metrics = pd.read_csv(DATA_DIR / "panelB_tp53_kras_metrics.tsv", sep="\t")

    # Order rows by progression from strongest-FOLFOX-favored to FOLFIRI-favored:
    # triple combo at top, then peel back features one at a time, ending with Both WT.
    desired_order = [
        "Both mut + 17p LOSS (TP53+/KRAS+/17p-)",
        "Both mut (TP53+ / KRAS+)",
        "TP53-only (TP53+ / KRAS-WT)",
        "KRAS-only (TP53-WT / KRAS+)",
        "Both WT (TP53-WT / KRAS-WT)",
    ]
    # Compact display labels for the y-axis
    short_labels = {
        "Both mut + 17p LOSS (TP53+/KRAS+/17p-)": r"TP53$+$ / KRAS$+$ / 17p$-$",
        "Both mut (TP53+ / KRAS+)":               r"TP53$+$ / KRAS$+$",
        "TP53-only (TP53+ / KRAS-WT)":            r"TP53$+$ / KRAS$-$",
        "KRAS-only (TP53-WT / KRAS+)":            r"TP53$-$ / KRAS$+$",
        "Both WT (TP53-WT / KRAS-WT)":            r"TP53$-$ / KRAS$-$",
    }
    # Filter to rows actually present (in case some old runs lack the 17p group)
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
        # Color by direction: FOLFOX-favoring red, FOLFIRI-favoring blue
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
    ax.set_xlabel("PFS HR (FOLFOX vs FOLFIRI), 95% CI", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics["display_label"].tolist(), fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_title("TP53 / KRAS co-mutation rule (binary)", fontsize=10, pad=8)

    # Right-side annotations
    x_text = 2.7
    ax.text(x_text, n_rows - 0.5, "n  /  P (Wald)",
              fontsize=8, va="bottom", ha="left", fontweight="bold")
    for i, row in metrics.iterrows():
        y = y_pos[i]
        ax.text(x_text, y, f"n={int(row['n']):>4}   P={row['PFS_P']:.2g}",
                  fontsize=8, va="center", ha="left", family="monospace")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "panelB_tp53_kras_forest.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "panelB_tp53_kras_forest.png", dpi=200,
                  bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG_DIR / 'panelB_tp53_kras_forest.pdf'} + .png")


def main():
    state = pickle.load(open(OUT / "attribution_matrix.pkl", "rb"))
    K_choice = sys.argv[1] if len(sys.argv) > 1 else "5"
    pmd_path = OUT / f"pmd_K{K_choice}_cv2.87.pkl"
    pmd = pickle.load(open(pmd_path, "rb"))
    res = pmd["res"]
    print(f"\n=== Using PMD fit: {pmd_path.name}  (K = {res['K']}) ===")
    print("\n=== Exporting panel data to panel_data/ ===")
    export_panel_a_data(state, res)
    export_panel_b_tp53_kras_data()
    print("\n=== Rendering preview figures to figure_panels/ ===")
    panel_a_signature_heatmap(state, res)
    panel_b_tp53_kras_forest()
    panel_c_triple_mutant_km()
    panel_d_double_wt_km()


if __name__ == "__main__":
    main()
