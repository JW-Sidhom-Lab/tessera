"""Variant- and CNA-level attribution of τ̂ for the predictive biomarker.

The full pipeline is end-to-end linear AFTER the per-patient mean/max aggregation:
    x_p (3716,)  →  raw_scaler  →  PCA  →  pcs_scaler
                 →  SparsePLS internal center+scale  →  β_PLS
                 →  sign-align  →  τ̂_p
We compose all post-aggregation transforms into one effective coefficient vector
β_x_eff (length 3716) such that  τ̂_p ≈ β_x_eff · x_p + const_eff  exactly.

We then unwind the per-patient aggregation to attribute τ̂_p to each
individual variant and CNA segment in patient p:

    SNV mean part: per-variant credit = β_x_eff[snv_mean_d] · ẽ_v[d] / n_p
    SNV max  part: per-variant credit only at dims d where v is argmax
                   = β_x_eff[snv_max_d] · ẽ_v[d] · 1[v = argmax_d in p]
    (ẽ_v is the RobustScaled variant embedding)

Mirror for CNAs via cna_mean / cna_max coefficients. Burden columns
(tmb_log1p, cna_log1p) get a single per-patient scalar contribution.

Reconstruction guarantee:
    Σ_v_snv attribution_pv  +  Σ_v_cna attribution_pv  +  burden_p  +  const_eff
        ≡  τ̂_p   for every p, up to float roundoff.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from .dr import _arm_mu_oof, _propensity_oof, interaction_test
from .spls import SparsePLS


# ============================================================================
# 1. Full-data refit - single SparsePLS τ̂ on the full cohort
# ============================================================================

def fit_full_data_model(
    df: pd.DataFrame, X: np.ndarray, *,
    pca_var: float = 0.99,
    K_mu1: int, K_mu0: int, K_e: int, K_tau: int,
    sp_mu1: float = 0.0, sp_mu0: float = 0.0,
    sp_e: float = 0.0, sp_tau: float = 0.0,
    n_inner: int = 5, seed: int = 42,
    horizon_months: float = 36.0,
    eps_e: float = 0.05,
) -> dict:
    """Re-fit the entire DR-learner pipeline on the FULL cohort (no outer CV).

    Inner-CV nuisance estimates μ̂_t and ê are still used to construct ψ_DR
    honestly, then a single SparsePLS τ̂ is fit on the full cohort's ψ_DR.
    Sign-align via the full-cohort τ × treatment interaction direction.

    Returns dict with raw_scaler, pca, pcs_scaler, pls_tau, sign, plus the
    in-sample τ̂ for reconstruction-checking.
    """
    raw_scaler = StandardScaler().fit(X)
    Xz = raw_scaler.transform(X)
    pca = PCA(n_components=pca_var, random_state=seed).fit(Xz)
    Xp = pca.transform(Xz)
    pcs_scaler = StandardScaler().fit(Xp)
    Xw = pcs_scaler.transform(Xp)

    mu1 = _arm_mu_oof(df, Xw, arm_value=1, K=K_mu1, sparsity=sp_mu1,
                          horizon=horizon_months, n_inner=n_inner, seed=seed)
    mu0 = _arm_mu_oof(df, Xw, arm_value=0, K=K_mu0, sparsity=sp_mu0,
                          horizon=horizon_months, n_inner=n_inner, seed=seed)
    e_in = _propensity_oof(df, Xw, K=K_e, sparsity=sp_e,
                              n_inner=n_inner, seed=seed)

    Y = df["pfs_t"].values
    T = df["arm"].values.astype(float)
    e_clip = np.clip(e_in, eps_e, 1 - eps_e)
    psi = (mu1 - mu0
           + T       * (Y - mu1) / e_clip
           - (1 - T) * (Y - mu0) / (1 - e_clip))

    pls_tau = SparsePLS(n_components=K_tau, sparsity=sp_tau).fit(Xw, psi)
    tau_raw = pls_tau.predict(Xw).ravel()

    # Sign-align via interaction direction (consistent with discovery scripts)
    itx = interaction_test(df, tau_raw, time_col="pfs_t", event_col="pfs_e")
    sign = -1.0 if itx["HR"] > 1.0 else +1.0
    tau_full = sign * tau_raw

    return dict(
        raw_scaler=raw_scaler,
        pca=pca,
        pcs_scaler=pcs_scaler,
        pls_tau=pls_tau,
        sign=sign,
        tau_full=tau_full,
        n_pca_components=int(Xw.shape[1]),
    )


# ============================================================================
# 2. Compose the linear chain into β_x_eff (3716,)
# ============================================================================

def build_effective_coefficients(model: dict) -> tuple[np.ndarray, float]:
    """Compose raw_scaler → PCA → pcs_scaler → SparsePLS → sign into a single
    affine map  τ̂_p = β_x_eff · x_p + const_eff  on the 3716-d input space.

    Verification: caller should compute τ̂ via this composed map and compare
    against model['tau_full'] - mismatch should be < 1e-6 (float roundoff).
    """
    raw   = model["raw_scaler"]
    pca   = model["pca"]
    pcs   = model["pcs_scaler"]
    spls  = model["pls_tau"]
    sign  = model["sign"]

    coef_pls = spls.coef_.ravel()                     # (K,)

    # === SparsePLS internal: τ_raw = ((v3 - x_mean) / x_std) @ coef + y_mean ===
    beta_v3  = coef_pls / spls._x_std                  # (K,)
    const_v3 = -float(spls._x_mean @ beta_v3) + float(spls._y_mean)

    # === pcs_scaler: v3 = (v2 - pcs.mean_) / pcs.scale_ ===
    beta_v2  = beta_v3 / pcs.scale_                    # (K,)
    const_v2 = -float(pcs.mean_ @ beta_v2) + const_v3

    # === sklearn PCA: v2 = (v1 - pca.mean_) @ components_.T ===
    beta_v1  = pca.components_.T @ beta_v2             # (3716,)
    const_v1 = -float(pca.mean_ @ beta_v1) + const_v2

    # === raw_scaler: v1 = (x - raw.mean_) / raw.scale_ ===
    beta_x   = beta_v1 / raw.scale_                    # (3716,)
    const_x  = -float(raw.mean_ @ beta_x) + const_v1

    # === sign alignment ===
    return sign * beta_x, sign * const_x


def reconstruction_check(model: dict, X_patients: np.ndarray) -> dict:
    """Reconstruct τ̂ via the composed linear map and compare to model.predict.

    Returns dict with max_abs_err, max_rel_err and a boolean pass flag.
    """
    beta, const = build_effective_coefficients(model)
    tau_recon = X_patients @ beta + const
    tau_true = model["tau_full"]
    abs_err = np.abs(tau_recon - tau_true)
    rel_err = abs_err / np.maximum(np.abs(tau_true), 1.0)
    return {
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
        "max_rel_err": float(rel_err.max()),
        "passes": bool(rel_err.max() < 1e-6),
    }


# ============================================================================
# 3. Block indices for the 3716-d patient feature vector
# ============================================================================

def feature_blocks(d_snv: int = 1169, d_cna: int = 688) -> dict:
    """Index ranges for snv_mean / snv_max / cna_mean / cna_max / burden."""
    snv_mean = (0, d_snv)
    snv_max  = (d_snv, 2 * d_snv)
    cna_mean = (2 * d_snv, 2 * d_snv + d_cna)
    cna_max  = (2 * d_snv + d_cna, 2 * d_snv + 2 * d_cna)
    tmb      = 2 * d_snv + 2 * d_cna
    cna_n    = 2 * d_snv + 2 * d_cna + 1
    return {"snv_mean": snv_mean, "snv_max": snv_max,
              "cna_mean": cna_mean, "cna_max": cna_max,
              "tmb": tmb, "cna_count": cna_n}


# ============================================================================
# 4. Per-(patient, variant) attribution
# ============================================================================

def per_variant_attributions(
    *,
    beta_x_eff: np.ndarray,
    snv_emb_scaled: np.ndarray,            # (N_snv_total, d_snv) RobustScaled
    cna_emb_scaled: np.ndarray,            # (N_cna_total, d_cna) RobustScaled
    snv_meta: pd.DataFrame,                # rows align to snv_emb_scaled
    cna_meta: pd.DataFrame,                # rows align to cna_emb_scaled
    patients: list[str],                   # patients to attribute (in cohort)
) -> pd.DataFrame:
    """Decompose τ̂_p into per-variant and per-CNA contributions.

    snv_meta must have a 'pid' column (9-char patient id).
    cna_meta must have a 'pid' column.

    Returns a long-format DataFrame:
      pid | source (snv|cna) | row_id | attribution
    Plus the relevant pass-through metadata for downstream gene aggregation.
    """
    blocks = feature_blocks(snv_emb_scaled.shape[1], cna_emb_scaled.shape[1])
    beta_snv_mean = beta_x_eff[blocks["snv_mean"][0]:blocks["snv_mean"][1]]
    beta_snv_max  = beta_x_eff[blocks["snv_max"][0]:blocks["snv_max"][1]]
    beta_cna_mean = beta_x_eff[blocks["cna_mean"][0]:blocks["cna_mean"][1]]
    beta_cna_max  = beta_x_eff[blocks["cna_max"][0]:blocks["cna_max"][1]]

    snv_pid_idx = snv_meta.groupby("pid").indices  # dict pid -> array of row idx
    cna_pid_idx = cna_meta.groupby("pid").indices

    rows = []
    pat_set = set(patients)
    for pid in patients:
        # ----- SNVs -----
        snv_idx = snv_pid_idx.get(pid, np.array([], dtype=int))
        if len(snv_idx) > 0:
            X_p = snv_emb_scaled[snv_idx]              # (n_snv_p, d_snv)
            n_snv_p = len(snv_idx)
            # Mean part: each variant gets its share of (β·ẽ_v / n_p)
            mean_contrib = (X_p @ beta_snv_mean) / n_snv_p   # (n_snv_p,)
            # Max part: the argmax variant at each dim gets β[d] · ẽ_v[d]
            argmax = np.argmax(X_p, axis=0)             # (d_snv,)
            d_idx = np.arange(X_p.shape[1])
            max_vals = X_p[argmax, d_idx]               # (d_snv,)
            per_dim_max = beta_snv_max * max_vals       # (d_snv,)
            max_contrib = np.zeros(n_snv_p)
            np.add.at(max_contrib, argmax, per_dim_max)
            attr = mean_contrib + max_contrib
            for i, ri in enumerate(snv_idx):
                rows.append((pid, "snv", int(ri), float(attr[i])))

        # ----- CNAs -----
        cna_idx = cna_pid_idx.get(pid, np.array([], dtype=int))
        if len(cna_idx) > 0:
            X_p = cna_emb_scaled[cna_idx]
            n_cna_p = len(cna_idx)
            mean_contrib = (X_p @ beta_cna_mean) / n_cna_p
            argmax = np.argmax(X_p, axis=0)
            d_idx = np.arange(X_p.shape[1])
            max_vals = X_p[argmax, d_idx]
            per_dim_max = beta_cna_max * max_vals
            max_contrib = np.zeros(n_cna_p)
            np.add.at(max_contrib, argmax, per_dim_max)
            attr = mean_contrib + max_contrib
            for i, ri in enumerate(cna_idx):
                rows.append((pid, "cna", int(ri), float(attr[i])))

    df = pd.DataFrame(rows, columns=["pid", "source", "row_id", "attribution"])
    return df


def per_patient_burden_contribution(
    beta_x_eff: np.ndarray, X_patients: np.ndarray, patients: list[str],
    d_snv: int = 1169, d_cna: int = 688,
) -> pd.DataFrame:
    """Burden contributions: tmb_log1p + cna_log1p (single scalar per patient)."""
    blocks = feature_blocks(d_snv, d_cna)
    tmb_idx = blocks["tmb"]
    cna_n_idx = blocks["cna_count"]
    tmb_contrib = beta_x_eff[tmb_idx] * X_patients[:, tmb_idx]
    cna_n_contrib = beta_x_eff[cna_n_idx] * X_patients[:, cna_n_idx]
    return pd.DataFrame({
        "pid": patients,
        "burden_tmb": tmb_contrib,
        "burden_cna_count": cna_n_contrib,
        "burden_total": tmb_contrib + cna_n_contrib,
    })


# ============================================================================
# 5. End-to-end attribution + reconstruction sanity test
# ============================================================================

def attribute_cohort(
    *,
    model: dict,
    X_patients: np.ndarray,                  # (n, 3716)
    patients: list[str],                     # length n
    tau_true: np.ndarray,                    # (n,) - model['tau_full'] sliced to these patients
    snv_emb_scaled: np.ndarray,
    cna_emb_scaled: np.ndarray,
    snv_meta: pd.DataFrame,
    cna_meta: pd.DataFrame,
) -> dict:
    """Compute per-(patient, variant) attributions for the supplied patients.

    Caller must pass `tau_true` (the model's prediction for these specific
    patients); this is used for the sum-of-attributions reconstruction check.
    """
    beta, const = build_effective_coefficients(model)

    var_attr = per_variant_attributions(
        beta_x_eff=beta,
        snv_emb_scaled=snv_emb_scaled,
        cna_emb_scaled=cna_emb_scaled,
        snv_meta=snv_meta, cna_meta=cna_meta,
        patients=patients,
    )
    burden = per_patient_burden_contribution(
        beta, X_patients, patients,
        d_snv=snv_emb_scaled.shape[1], d_cna=cna_emb_scaled.shape[1],
    )

    # Reconstruction by attribution: sum per-patient should match tau_true
    snv_sum = var_attr[var_attr["source"] == "snv"].groupby("pid")["attribution"].sum()
    cna_sum = var_attr[var_attr["source"] == "cna"].groupby("pid")["attribution"].sum()
    burden_d = burden.set_index("pid")
    tau_recon_attr = (snv_sum.reindex(patients).fillna(0).values
                       + cna_sum.reindex(patients).fillna(0).values
                       + burden_d["burden_total"].reindex(patients).fillna(0).values
                       + const)
    abs_err = np.abs(tau_recon_attr - tau_true)
    rel_err = abs_err / np.maximum(np.abs(tau_true), 1.0)
    return dict(
        beta_x_eff=beta,
        const_eff=const,
        attr_recon_max_abs_err=float(abs_err.max()),
        attr_recon_max_rel_err=float(rel_err.max()),
        attr_recon_passes=bool(rel_err.max() < 1e-4),  # tolerant of float roundoff over 100s of summed contributions
        variant_attributions=var_attr,
        burden=burden,
    )
