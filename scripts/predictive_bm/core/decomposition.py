"""Sparse low-rank decomposition of the per-(patient, gene) attribution matrix.

Implements Penalized Matrix Decomposition (PMD) of Witten, Tibshirani, and
Hastie (Biostatistics 2009; doi:10.1093/biostatistics/kxp008) - the L1-on-rows-
and-L1-on-columns variant, denoted PMD(L1, L1) - directly in NumPy, with no R
dependency.

For a signed matrix A ∈ R^{P × G}, PMD(L1, L1) finds a rank-K approximation
A ≈ U D V^T by sequentially solving, for each component k on the deflated
residual matrix A_k:

    max_{u, v}   u^T A_k v
    subject to   ||u||_2 ≤ 1,  ||v||_2 ≤ 1,
                 ||u||_1 ≤ c_u, ||v||_1 ≤ c_v

The optimization alternates between soft-thresholded updates for u and v,
binary-searching the soft-threshold λ to enforce the L1 budget. At
convergence, d_k = u^T A_k v, and A_{k+1} = A_k − d_k u v^T.

Without penalties (c_u = c_v = √dim) this reduces to the standard SVD.
With penalties, the components lose mutual orthogonality (a feature, not a
bug, since biological signals are not orthogonal in any natural basis).

References:
- Witten, Tibshirani, Hastie (2009) *Biostatistics* 10(3):515–534.
- Witten and Tibshirani (2009) *Stat. Appl. Genet. Mol. Biol.* 8(1):28
  (sparse CCA extension; not implemented here).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft-threshold: sign(x) * max(|x| - lam, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _binary_search_lambda(
    x: np.ndarray, c: float, *, max_iter: int = 100, tol: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Find the smallest λ ≥ 0 such that ||S(x, λ) / ||S(x, λ)||_2||_1 ≤ c.

    If the L2-normalized soft-thresholded x at λ=0 already has L1 ≤ c,
    return it with λ = 0 (no thresholding needed). Otherwise binary search.

    Returns (S(x, λ) normalized to unit L2, λ).
    """
    abs_x = np.abs(x)
    # L1 budget c must lie in [1, sqrt(p)]; outside this range the constraint
    # is either always active or never active.
    p = x.size
    if c >= np.sqrt(p):
        # No effective constraint
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return np.zeros_like(x), 0.0
        return x / norm, 0.0

    # Check λ=0
    norm0 = np.linalg.norm(x)
    if norm0 < 1e-12:
        return np.zeros_like(x), 0.0
    u0 = x / norm0
    if np.sum(np.abs(u0)) <= c:
        return u0, 0.0

    # Binary search: find λ such that L1 of normalized soft-threshold = c
    lo, hi = 0.0, abs_x.max()
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = _soft_threshold(x, mid)
        n = np.linalg.norm(s)
        if n < 1e-12:
            hi = mid
            continue
        l1 = np.sum(np.abs(s)) / n
        if l1 > c:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    s = _soft_threshold(x, hi)
    n = np.linalg.norm(s)
    if n < 1e-12:
        return np.zeros_like(x), hi
    return s / n, hi


def _pmd_rank1(
    A: np.ndarray, c_u: float, c_v: float, *,
    max_iter: int = 200, tol: float = 1e-6, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Single rank-1 PMD(L1, L1) component. Returns (u, v, d, n_iter).

    Initialized at the leading singular vectors of A (truncated SVD on
    a thin matrix is fast even for 1500x500).
    """
    P, G = A.shape
    U_init, _, Vt_init = svd(A, full_matrices=False)
    u = U_init[:, 0].copy()
    v = Vt_init[0, :].copy()
    if np.sum(u) < 0:
        u = -u
        v = -v

    for it in range(max_iter):
        u_old = u.copy()
        v_old = v.copy()

        # Update v given u: maximize u^T A v subject to v constraints
        # gradient direction is A^T u
        v_grad = A.T @ u
        v, _ = _binary_search_lambda(v_grad, c_v)

        # Update u given v
        u_grad = A @ v
        u, _ = _binary_search_lambda(u_grad, c_u)

        if (np.linalg.norm(u - u_old) < tol and
            np.linalg.norm(v - v_old) < tol):
            break

    d = float(u @ A @ v)
    return u, v, d, it + 1


def pmd_l1l1(
    A: np.ndarray, K: int, *,
    c_u: float | None = None,
    c_v: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """Rank-K PMD(L1, L1) on signed matrix A ∈ R^{P x G}.

    Parameters
    ----------
    A : (P, G) numpy array, signed.
    K : number of components.
    c_u : L1 budget for left vectors (patient loadings). Range: [1, sqrt(P)].
          Default: sqrt(P) (no effective penalty on patient loadings -
          matching the conventional choice of leaving u dense).
    c_v : L1 budget for right vectors (gene loadings). Range: [1, sqrt(G)].
          Default: sqrt(G) / 2 (moderate sparsity on signatures).
    max_iter : per-component iteration cap.
    tol : convergence tolerance on ||u_new - u_old|| + ||v_new - v_old||.
    seed : reserved for future stochastic init; current init is deterministic SVD.

    Returns
    -------
    dict with:
      U : (P, K) left vectors (patient loadings, unit L2 norm)
      D : (K,) singular values (signs absorbed into U/V)
      V : (G, K) right vectors (gene loadings, unit L2 norm, sparse)
      n_iter : (K,) iterations to convergence per component
      reconstruction : (P, G) U @ diag(D) @ V^T
      explained_var : float - ||reconstruction||_F^2 / ||A||_F^2
    """
    P, G = A.shape
    if c_u is None:
        c_u = float(np.sqrt(P))
    if c_v is None:
        c_v = float(np.sqrt(G) / 2.0)

    U = np.zeros((P, K))
    V = np.zeros((G, K))
    D = np.zeros(K)
    n_iter = np.zeros(K, dtype=int)

    A_resid = A.copy()
    for k in range(K):
        u, v, d, ni = _pmd_rank1(A_resid, c_u, c_v,
                                   max_iter=max_iter, tol=tol, seed=seed + k)
        if abs(d) < 1e-12:
            # No more signal; truncate
            K = k
            U = U[:, :K]; V = V[:, :K]; D = D[:K]; n_iter = n_iter[:K]
            break
        U[:, k] = u
        V[:, k] = v
        D[k]    = d
        n_iter[k] = ni
        A_resid = A_resid - d * np.outer(u, v)

    recon = U @ np.diag(D) @ V.T
    A_norm = float(np.linalg.norm(A))
    R_norm = float(np.linalg.norm(recon))
    expl_var = (R_norm ** 2) / max(A_norm ** 2, 1e-12)

    return dict(
        U=U, D=D, V=V,
        n_iter=n_iter,
        reconstruction=recon,
        explained_var=expl_var,
        c_u=c_u,
        c_v=c_v,
        K=int(U.shape[1]),
    )


# ============================================================================
# Bootstrap-Procrustes stability for K selection
# ============================================================================

def procrustes_align(V_ref: np.ndarray, V_boot: np.ndarray) -> tuple[np.ndarray, float]:
    """Align V_boot to V_ref by orthogonal Procrustes.

    Returns (V_boot_aligned, mean_canonical_corr).
    """
    if V_ref.shape != V_boot.shape:
        raise ValueError(f"V_ref {V_ref.shape} vs V_boot {V_boot.shape}")
    M = V_ref.T @ V_boot
    Ur, _, Vrt = svd(M, full_matrices=False)
    R = Ur @ Vrt
    V_aligned = V_boot @ R.T
    K = V_ref.shape[1]
    # Per-component canonical correlation (mean cosine) after alignment
    corrs = []
    for k in range(K):
        a = V_ref[:, k]
        b = V_aligned[:, k]
        n_a = np.linalg.norm(a)
        n_b = np.linalg.norm(b)
        if n_a < 1e-12 or n_b < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(abs(float(a @ b) / (n_a * n_b)))
    return V_aligned, float(np.mean(corrs))


def bootstrap_stability(
    A: np.ndarray, K: int, *,
    n_boot: int = 100, c_u: float | None = None, c_v: float | None = None,
    seed: int = 42,
) -> dict:
    """Bootstrap-Procrustes stability of PMD components at rank K.

    Resamples patients (rows of A) with replacement, refits PMD, aligns
    bootstrapped V to reference V via Procrustes, returns the distribution
    of mean canonical correlations.

    Returns dict with mean stability, per-bootstrap stabilities, and the
    reference V.
    """
    P, G = A.shape
    rng = np.random.default_rng(seed)

    ref = pmd_l1l1(A, K, c_u=c_u, c_v=c_v)
    V_ref = ref["V"]

    stabilities = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, P, size=P)
        A_b = A[idx]
        boot = pmd_l1l1(A_b, K, c_u=c_u, c_v=c_v)
        if boot["V"].shape[1] != K:
            # Bootstrap fit collapsed; pad
            stabilities[b] = 0.0
            continue
        _, sim = procrustes_align(V_ref, boot["V"])
        stabilities[b] = sim

    return dict(
        K=K,
        mean_stability=float(stabilities.mean()),
        median_stability=float(np.median(stabilities)),
        per_boot=stabilities,
        V_ref=V_ref,
    )


def occupancy_normalize(A: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Divide each column by sqrt(nonzero count) so per-event signal is on the
    same scale across columns regardless of occupancy.

    Column std under the model A_pg = m_g * Bernoulli(occupancy_g) is roughly
    m_g * sqrt(occupancy_g). Dividing by sqrt(K_g) where K_g = sum(|A| > eps)
    gives columns with std m_g / sqrt(N), which is constant when per-event
    magnitudes are comparable (which we confirmed empirically: median per-event
    |A| is ~22-24 for both SNV and CNA columns).

    Returns (A_normalized, scale_factors) where scale_factors[j] = sqrt(K_j).
    To map a normalized loading v_norm back to original-scale, multiply by
    scale_factors. (Loadings in normalized space measure "per-event importance";
    loadings in original space measure "total contribution to tau_hat".)
    """
    K = (np.abs(A) > eps).sum(axis=0).astype(float)
    K_safe = np.maximum(K, 1.0)
    scale = np.sqrt(K_safe)
    return A / scale[np.newaxis, :], scale


def select_K_by_stability(
    A: np.ndarray, *,
    K_grid: list[int] | None = None,
    n_boot: int = 50,
    c_v: float | None = None,
    target_stability: float = 0.7,
    seed: int = 42,
) -> dict:
    """Sweep K, return stability per K and recommended K.

    Recommended K = smallest K in K_grid with mean stability >= target.
    """
    if K_grid is None:
        K_grid = [3, 4, 5, 6, 8, 10]

    out = {}
    for K in K_grid:
        s = bootstrap_stability(A, K, n_boot=n_boot, c_v=c_v, seed=seed)
        out[K] = s
        print(f"  K={K}  mean stability={s['mean_stability']:.3f}", flush=True)

    # Pick smallest K with stability >= target
    candidates = [K for K in K_grid if out[K]["mean_stability"] >= target_stability]
    K_rec = candidates[0] if candidates else max(K_grid, key=lambda K: out[K]["mean_stability"])
    return dict(
        per_K=out,
        recommended_K=K_rec,
        target=target_stability,
    )
