"""PDAC predictive-treatment-effect signature decomposition (FOLFIRINOX vs Gem/Abraxane).

Pipeline mirrors crc_signatures.py:
  1. Load patient features and PDAC cohort (stage IV, first-line).
  2. Fit full-cohort DR-learner; compose beta_eff.
  3. Compute per-(patient, variant/CNA) attribution.
  4. Aggregate to per-(patient, gene-or-arm) matrix A.
  5. PMD(L1, L1) decomposition. K defaults to 5; pass ``--K 10`` for
     the manuscript Sup Fig 12 configuration.
  6. Save attribution matrix and decomposition results.

Outputs:
  attribution_analysis/pdac_signatures/
    attribution_matrix.pkl     (P x G + metadata)
    pmd_K{K}_cv{c_v}.pkl
    top_genes_per_signature.tsv
"""
from __future__ import annotations

import sys
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.features import build_patient_features
from core.cohorts import build_pdac_met
from core.attribution import (
    fit_full_data_model,
    build_effective_coefficients,
    reconstruction_check,
    attribute_cohort,
)
from core.arm_mapping import (
    overlap_segments_with_arms,
    attach_arms_to_attributions,
    feature_attribution_per_patient,
    compute_arm_direction_map,
    label_signature_feature,
)
from core.decomposition import pmd_l1l1


DATA = ROOT / ".." / ".." / "data" / "msk_chord_2024"
GT = DATA / "GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv"
RAW_PKL = ROOT / "msk_chord_latent_features_raw.pkl"
CACHE = ROOT / "cache" / "patient_features.pkl"
OUT_DIR = ROOT / "attribution_analysis" / "pdac_signatures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PDAC validated DR-learner config (from pdac_discovery.py: PCA(0.99) coord-descent)
PCA_VAR = 0.99
K_MU1, SP_MU1 = 2, 0.5
K_MU0, SP_MU0 = 3, 0.85
K_E,   SP_E   = 2, 0.0
K_TAU, SP_TAU = 5, 0.85
N_INNER = 5
SEED    = 42
HORIZON = 36.0

# Signature decomposition config
K_SIG = 5  # default rank; pass --K to override


def banner(title):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def build_attribution_matrix(*, force: bool = False):
    """Build the per-(patient, feature) attribution matrix A for PDAC.

    Caches at OUT_DIR/attribution_matrix.pkl. Set force=True to rebuild.
    """
    cache_path = OUT_DIR / "attribution_matrix.pkl"
    if cache_path.exists() and not force:
        print(f"  loading cached attribution matrix from {cache_path}", flush=True)
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    banner("Step 1 - load patient features")
    feat = build_patient_features(RAW_PKL, cache=CACHE)
    print(f"  patient features: {feat.shape}")

    banner("Step 2 - build PDAC stage IV first-line cohort")
    df, X = build_pdac_met(GT, feat, horizon_months=HORIZON)
    arm_counts = df["arm"].value_counts().to_dict()
    print(f"  cohort n={len(df)}")
    print(f"  arm counts: FOLFIRINOX(arm=1)={arm_counts.get(1,0)}  "
          f"GemAbraxane(arm=0)={arm_counts.get(0,0)}")
    print(f"  PFS events: {int(df['pfs_e'].sum())}/{len(df)}")
    print(f"  X shape: {X.shape}")

    banner("Step 3 - fit full-cohort DR-learner (no outer CV)")
    t1 = time.time()
    model = fit_full_data_model(
        df, X,
        pca_var=PCA_VAR,
        K_mu1=K_MU1, K_mu0=K_MU0, K_e=K_E, K_tau=K_TAU,
        sp_mu1=SP_MU1, sp_mu0=SP_MU0, sp_e=SP_E, sp_tau=SP_TAU,
        n_inner=N_INNER, seed=SEED, horizon_months=HORIZON,
    )
    print(f"  fit done in {time.time()-t1:.1f}s; PCA components={model['n_pca_components']}")

    banner("Step 4 - reconstruction unit test")
    rc = reconstruction_check(model, X)
    print(f"  max rel err = {rc['max_rel_err']:.3e}; passes = {rc['passes']}")
    assert rc["passes"], "Reconstruction unit test failed - investigate before proceeding"

    banner("Step 5 - load raw embeddings and per-variant attribution")
    with open(RAW_PKL, "rb") as fh:
        bundle = pickle.load(fh)
    snv_emb = bundle["variant_features"]
    cna_emb = bundle["cna_features"]
    snv_meta = bundle["data_snv"].copy()
    cna_meta = bundle["data_cna"].copy()
    snv_meta["pid"] = snv_meta["Tumor_Sample_Barcode"].str[:9]
    cna_meta["pid"] = cna_meta["Tumor_Sample_Barcode"].str[:9]
    snv_z = RobustScaler().fit_transform(snv_emb)
    cna_z = RobustScaler().fit_transform(cna_emb)

    patients = list(df["pid"].values)
    out = attribute_cohort(
        model=model,
        X_patients=X, patients=patients,
        tau_true=model["tau_full"],
        snv_emb_scaled=snv_z, cna_emb_scaled=cna_z,
        snv_meta=snv_meta, cna_meta=cna_meta,
    )
    print(f"  attribution rows: {len(out['variant_attributions'])}")
    print(f"  attribution recon max rel err: {out['attr_recon_max_rel_err']:.3e}")

    banner("Step 6 - feature-level aggregation (genes for SNV, arms for CNA)")
    cna_meta_with_rid = cna_meta.copy()
    cna_meta_with_rid["row_id"] = cna_meta_with_rid.index
    cna_for_overlap = cna_meta_with_rid.rename(
        columns={"Chromosome": "Chromosome", "Start": "Start", "End": "End"})
    cna_to_arm = overlap_segments_with_arms(cna_for_overlap)
    print(f"  cna_to_arm: {len(cna_to_arm)} (segment, arm) pairs")
    attr_with_feature = attach_arms_to_attributions(
        out["variant_attributions"], snv_meta, cna_meta_with_rid, cna_to_arm)
    feat_attr = feature_attribution_per_patient(attr_with_feature)
    print(f"  per-(patient, feature) rows: {len(feat_attr)}")

    features = sorted(feat_attr["feature"].dropna().unique().tolist())
    pid_to_idx = {p: i for i, p in enumerate(patients)}
    feat_to_idx = {f: j for j, f in enumerate(features)}
    A = np.zeros((len(patients), len(features)), dtype=float)
    for pid, feature, attrib in zip(feat_attr["pid"].values,
                                       feat_attr["feature"].values,
                                       feat_attr["attribution"].values):
        if isinstance(feature, float) and np.isnan(feature):
            continue
        if pid not in pid_to_idx or feature not in feat_to_idx:
            continue
        A[pid_to_idx[pid], feat_to_idx[feature]] = attrib

    # Identify which columns are arm labels
    ARM_LABELS = set()
    for c in [str(i) for i in range(1, 23)] + ["X", "Y"]:
        ARM_LABELS.add(f"{c}p"); ARM_LABELS.add(f"{c}q")
    is_arm = np.array([f in ARM_LABELS for f in features])

    print(f"  attribution matrix A: {A.shape}")
    print(f"    SNV-gene columns:  {(~is_arm).sum()}")
    print(f"    CNA-arm columns:   {is_arm.sum()}")
    print(f"  density (non-zero entries): {(A != 0).mean():.3f}")
    print(f"  Frobenius norm: {np.linalg.norm(A):.2f}")

    banner("Step 7 - arm direction map (AMP vs LOSS at high attribution)")
    arm_dir = compute_arm_direction_map(A, features, is_arm, cna_meta, patients)
    print(f"  resolved {sum(1 for v in arm_dir.values() if v['sign'] != '?')} of "
          f"{is_arm.sum()} arms")
    sorted_arms = sorted(arm_dir.items(), key=lambda kv: -abs(kv[1]['rho']))
    for arm, info in sorted_arms[:10]:
        print(f"    {arm:>4}  rho={info['rho']:+.3f}  sign={info['sign']}  P={info['P']:.1e}")

    result = {
        "A": A,
        "patients": patients,
        "features": features,
        "is_arm": is_arm,
        "arm_dir": arm_dir,
        "genes": features,

        "df": df.reset_index(drop=True),
        "tau_full": model["tau_full"],
        "tau0": float(np.median(model["tau_full"])),
        "beta_eff": out["beta_x_eff"],
        "const_eff": out["const_eff"],
        "model_meta": {
            "n_pca_components": model["n_pca_components"],
            "sign": model["sign"],
        },
        "attr_recon_max_rel_err": out["attr_recon_max_rel_err"],
    }
    with open(cache_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"  saved to {cache_path}")
    return result


def run_pmd(state, *, K: int = 5):
    """Single PMD fit at K with c_v = sqrt(G)/8."""
    A = state["A"]
    P, G = A.shape
    state["scale_factors"] = np.ones(G)
    c_v = float(np.sqrt(G) / 8.0)

    banner(f"PMD(K={K}, c_v={c_v:.2f})")
    t0 = time.time()
    res = pmd_l1l1(A, K=K, c_v=c_v)
    elapsed = time.time() - t0
    nz = (np.abs(res["V"]) > 1e-10).sum(axis=0)
    print(f"  fit in {elapsed:.1f}s; expl_var={res['explained_var']:.3f}; "
          f"D={[f'{d:.2f}' for d in res['D']]}; nonzero/sig={nz.tolist()}")

    is_arm = state["is_arm"]
    arm_dir = state.get("arm_dir")
    summary_rows = []
    for k in range(res["K"]):
        v = res["V"][:, k]
        order = np.argsort(-np.abs(v))
        top_idx = order[:15]
        top = [(state["features"][i], float(v[i]), is_arm[i])
                 for i in top_idx if abs(v[i]) > 1e-10]
        top_str = "  ".join(label_signature_feature(f, w, ia, arm_dir)
                              for f, w, ia in top[:8])
        sign_total = float(np.sign(v.sum()))
        print(f"  sig {k} (D={res['D'][k]:.2f}, sum_dir={sign_total:+.0f}): {top_str}")
        for rank, (f, w, ia) in enumerate(top, start=1):
            bio_label = label_signature_feature(f, w, ia, arm_dir)
            summary_rows.append(dict(
                c_v=c_v, K=K, sig=k, D=res["D"][k],
                rank=rank, feature=f, kind="arm" if ia else "gene",
                bio_label=bio_label,
                loading=w, sig_direction=sign_total,
                n_nonzero=int(nz[k]), explained_var=res["explained_var"],
            ))

    fname = f"pmd_K{K}_cv{c_v:.2f}.pkl".replace(" ", "")
    with open(OUT_DIR / fname, "wb") as fh:
        pickle.dump(dict(res=res, c_v=c_v, K=K), fh)
    print(f"  saved to {OUT_DIR / fname}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "top_genes_per_signature.tsv", sep="\t", index=False)
    print(f"\nSummary table → {OUT_DIR / 'top_genes_per_signature.tsv'}")


def main():
    state = build_attribution_matrix(force=("--force" in sys.argv))
    K = K_SIG
    if "--K" in sys.argv:
        K = int(sys.argv[sys.argv.index("--K") + 1])
    run_pmd(state, K=K)


if __name__ == "__main__":
    main()
