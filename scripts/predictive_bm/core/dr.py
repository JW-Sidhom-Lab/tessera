"""DR-learner (doubly-robust / AIPW) HTE estimator with PLS regressors.

Pseudo-outcome:
    psi_DR = mu1(X) - mu0(X)
             + T*(Y - mu1(X))/e(X)
             - (1-T)*(Y - mu0(X))/(1 - e(X))

For survival with horizon h, mu_a(X) is the RMST_h prediction from a
Cox-PLS model fit on arm a only.

Cross-fitting structure (per outer fold, on outer-train set otr):
  - mu_1: PLS-Cox fit cross-validated on otr ∩ {arm=1}; for otr ∩ {arm=0}
    a single PLS-Cox fit on all otr ∩ {arm=1} (independent for non-arm
    patients) is used.
  - mu_0: mirror of above on the arm=0 subset.
  - e   : PLS-Logistic, OOF cross-validated on full otr.
The DR pseudo-outcome psi_DR is then a per-patient quantity on otr; a
separate sparse-PLS regression of psi_DR on X is fit on otr and used to
predict tau_hat on the held-out outer-test set.

Outer CV is StratifiedKFold over the (arm, event) joint label so each
fold is balanced on both treatment and outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .spls import SparsePLS


def _rmst_predict(cph: CoxPHFitter, Z_pred: pd.DataFrame, horizon: float,
                    dt: float = 0.5) -> np.ndarray:
    """Trapezoid-integrate predicted survival to compute RMST at horizon.

    Grid starts at t=0 (S(0)=1) and steps by dt to horizon inclusive, matching
    the validated grid used in the prior pipeline.
    """
    grid = np.arange(0.0, horizon + dt, dt)
    sf = cph.predict_survival_function(Z_pred, times=grid)
    return np.trapz(sf.values, grid, axis=0)


def _arm_mu_oof(df: pd.DataFrame, X: np.ndarray, *,
                  arm_value: int, K: int, sparsity: float,
                  horizon: float, n_inner: int, seed: int,
                  cox_pen: float = 0.05) -> np.ndarray:
    """Out-of-fold mu_arm(X) for ALL patients in df.

    For patients with arm == arm_value, OOF prediction via inner k-fold on
    those patients only. For patients with arm != arm_value, single
    PLS-Cox fit on all in-arm patients (independent for non-arm patients).
    """
    n = len(df)
    in_mask = (df["arm"].values == arm_value)
    in_pos  = np.where(in_mask)[0]
    out_pos = np.where(~in_mask)[0]

    arm_df = df.iloc[in_pos].reset_index(drop=True)
    arm_X  = X[in_pos]
    pred = np.full(n, np.nan, dtype=float)

    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, te in inner.split(arm_df, arm_df["pfs_e"]):
        Y_tr = arm_df.iloc[tr]["pfs_t"].values
        E_tr = arm_df.iloc[tr]["pfs_e"].values
        spls = SparsePLS(n_components=K, sparsity=sparsity).fit(arm_X[tr], Y_tr)
        Z_tr = spls.transform(arm_X[tr])
        Z_te = spls.transform(arm_X[te])
        Kused = Z_tr.shape[1]
        if Kused == 0:
            # PLS produced no components - fallback: predict cohort mean RMST
            # via Kaplan-Meier-ish proxy = mean Y_tr (rough but stable)
            pred[in_pos[te]] = np.mean(Y_tr)
            continue
        cols = [f"z{i}" for i in range(Kused)]
        df_tr = pd.DataFrame(Z_tr, columns=cols)
        df_tr["pfs_t"] = Y_tr
        df_tr["pfs_e"] = E_tr
        cph = CoxPHFitter(penalizer=cox_pen).fit(df_tr,
            duration_col="pfs_t", event_col="pfs_e")
        pred[in_pos[te]] = _rmst_predict(cph,
            pd.DataFrame(Z_te, columns=cols), horizon)

    if len(out_pos) > 0:
        spls = SparsePLS(n_components=K, sparsity=sparsity).fit(
            arm_X, arm_df["pfs_t"].values)
        Z_all   = spls.transform(arm_X)
        Z_other = spls.transform(X[out_pos])
        Kused = Z_all.shape[1]
        if Kused == 0:
            pred[out_pos] = float(np.mean(arm_df["pfs_t"].values))
        else:
            cols = [f"z{i}" for i in range(Kused)]
            df_all = pd.DataFrame(Z_all, columns=cols)
            df_all["pfs_t"] = arm_df["pfs_t"].values
            df_all["pfs_e"] = arm_df["pfs_e"].values
            cph = CoxPHFitter(penalizer=cox_pen).fit(df_all,
                duration_col="pfs_t", event_col="pfs_e")
            pred[out_pos] = _rmst_predict(cph,
                pd.DataFrame(Z_other, columns=cols), horizon)
    return pred


def _propensity_oof(df: pd.DataFrame, X: np.ndarray, *,
                       K: int, sparsity: float,
                       n_inner: int, seed: int) -> np.ndarray:
    """OOF propensity P(arm=1 | X) via PLS-Logistic with inner CV.

    Stratification: by EVENT (pfs_e), not by treatment, to match the
    validated pipeline. This keeps fold balance on outcome rather than on
    the variable being modeled.
    """
    n = len(df)
    Y = df["arm"].values.astype(int)
    strat = df["pfs_e"].values
    pred = np.full(n, np.nan, dtype=float)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, te in inner.split(X, strat):
        spls = SparsePLS(n_components=K, sparsity=sparsity).fit(
            X[tr], Y[tr].astype(float))
        Z_tr = spls.transform(X[tr])
        Z_te = spls.transform(X[te])
        if Z_tr.shape[1] == 0:
            pred[te] = float(Y[tr].mean())
            continue
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                  random_state=seed).fit(Z_tr, Y[tr])
        pred[te] = lr.predict_proba(Z_te)[:, 1]
    return pred


def fit_dr_learner(
    df: pd.DataFrame, X: np.ndarray, *,
    pca_var: float = 0.99,
    K_mu1: int, K_mu0: int, K_e: int, K_tau: int,
    sp_mu1: float = 0.0, sp_mu0: float = 0.0,
    sp_e: float = 0.0, sp_tau: float = 0.0,
    n_outer: int = 10, n_inner: int = 5, seed: int = 42,
    horizon_months: float = 36.0,
    eps_e: float = 0.05,
) -> dict:
    """Nested-CV DR-learner. Returns dict with tau_oof and preprocessing chain.

    Pipeline: StandardScaler -> PCA(pca_var) -> StandardScaler -> DR-learner.
    Each piece of the supervised DR-learner is held out per the inner k-fold
    so psi_DR is constructed from out-of-sample nuisance estimates.
    """
    n = len(df)

    # Preprocessing chain (fit on full cohort - unsupervised, mild leakage)
    raw_scaler = StandardScaler().fit(X)
    Xz = raw_scaler.transform(X)
    pca = PCA(n_components=pca_var, random_state=seed).fit(Xz)
    Xp = pca.transform(Xz)
    pcs_scaler = StandardScaler().fit(Xp)
    Xw = pcs_scaler.transform(Xp)

    strat = (df["arm"].astype(str) + "|" + df["pfs_e"].astype(str)).values
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    tau_oof = np.full(n, np.nan, dtype=float)
    for fold_id, (otr, ote) in enumerate(outer.split(df, strat), start=1):
        otr_df = df.iloc[otr].reset_index(drop=True)
        X_otr = Xw[otr]
        X_ote = Xw[ote]
        s_fold = seed + fold_id

        mu1 = _arm_mu_oof(otr_df, X_otr, arm_value=1, K=K_mu1,
                            sparsity=sp_mu1, horizon=horizon_months,
                            n_inner=n_inner, seed=s_fold)
        mu0 = _arm_mu_oof(otr_df, X_otr, arm_value=0, K=K_mu0,
                            sparsity=sp_mu0, horizon=horizon_months,
                            n_inner=n_inner, seed=s_fold)
        e_in = _propensity_oof(otr_df, X_otr, K=K_e, sparsity=sp_e,
                                  n_inner=n_inner, seed=s_fold)

        Y = otr_df["pfs_t"].values
        T = otr_df["arm"].values.astype(float)
        e_clip = np.clip(e_in, eps_e, 1 - eps_e)
        psi = (mu1 - mu0
               + T       * (Y - mu1) / e_clip
               - (1 - T) * (Y - mu0) / (1 - e_clip))

        tau_pls = SparsePLS(n_components=K_tau, sparsity=sp_tau).fit(X_otr, psi)
        tau_oof[ote] = tau_pls.predict(X_ote).ravel()

    return dict(
        tau_oof=tau_oof,
        raw_scaler=raw_scaler,
        pca=pca,
        pcs_scaler=pcs_scaler,
        n_pca_components=int(Xw.shape[1]),
    )


def interaction_test(df: pd.DataFrame, tau: np.ndarray, *,
                       time_col: str, event_col: str) -> dict:
    """Interaction Cox: time, event ~ tau + arm + tau*arm. Returns interaction HR/CI/P."""
    work = pd.DataFrame({
        time_col:  df[time_col].values,
        event_col: df[event_col].values,
        "tau":     tau,
        "T":       df["arm"].values.astype(int),
    })
    work["tauT"] = work["tau"] * work["T"]
    cph = CoxPHFitter().fit(work[[time_col, event_col, "tau", "T", "tauT"]],
                              duration_col=time_col, event_col=event_col)
    HR    = float(np.exp(cph.params_["tauT"]))
    HR_lo = float(np.exp(cph.confidence_intervals_.loc["tauT", "95% lower-bound"]))
    HR_hi = float(np.exp(cph.confidence_intervals_.loc["tauT", "95% upper-bound"]))
    P     = float(cph.summary.loc["tauT", "p"])
    return {
        "HR": HR, "HR_lo": HR_lo, "HR_hi": HR_hi, "P": P,
        "beta_T": float(cph.params_["T"]),
        "beta_int": float(cph.params_["tauT"]),
    }


def threshold_indifference(itx: dict) -> float:
    """Indifference τ̂ where the per-patient treatment effect crosses zero."""
    return -itx["beta_T"] / itx["beta_int"]


def per_group_arm_hr(df: pd.DataFrame, mask: np.ndarray, *,
                       time_col: str, event_col: str) -> dict:
    """Within a subgroup, fit Cox(time, event ~ arm). Returns HR/CI/P."""
    sub = df.loc[mask, [time_col, event_col, "arm"]].rename(columns={"arm": "T"})
    cph = CoxPHFitter().fit(sub[[time_col, event_col, "T"]],
                              duration_col=time_col, event_col=event_col)
    HR    = float(np.exp(cph.params_["T"]))
    HR_lo = float(np.exp(cph.confidence_intervals_.loc["T", "95% lower-bound"]))
    HR_hi = float(np.exp(cph.confidence_intervals_.loc["T", "95% upper-bound"]))
    P     = float(cph.summary.loc["T", "p"])
    return {"n": int(mask.sum()), "HR": HR, "HR_lo": HR_lo, "HR_hi": HR_hi, "P": P}
