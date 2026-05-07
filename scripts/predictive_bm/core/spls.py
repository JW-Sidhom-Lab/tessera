"""Sparse PLS via NIPALS with hard-thresholded weight vectors.

Standard PLS-NIPALS produces dense X-weights w_k that select linear
combinations of all features. Sparse PLS hard-thresholds w_k after each
NIPALS iteration, keeping only the top (1 - sparsity) fraction of
features by absolute weight. This yields per-component feature selection
with the latent-direction structure of PLS.

Sklearn-compatible interface: fit(X, y) -> self; transform(X) -> scores;
predict(X) -> Y_hat. Handles both univariate and multi-output y.

Notes
-----
- For univariate y, NIPALS converges in essentially 1 iteration (the
  y-block doesn't update).
- After thresholding, w is re-normalized so deflation behaves correctly.
- Components that collapse to zero weight after thresholding are dropped
  (the truncated weight matrix is kept). This can mean K_actual < K_requested.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def _topk_threshold(v: np.ndarray, sparsity: float) -> np.ndarray:
    """Keep top (1-sparsity) fraction by |v|, zero the rest."""
    if sparsity <= 0:
        return v
    p = v.size
    keep = max(1, int(round((1.0 - sparsity) * p)))
    if keep >= p:
        return v
    abs_v = np.abs(v)
    cutoff = np.partition(abs_v, p - keep)[p - keep]
    out = np.where(abs_v >= cutoff, v, 0.0)
    return out


class SparsePLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components: int = 2, sparsity: float = 0.0,
                 max_iter: int = 100, tol: float = 1e-6):
        self.n_components = n_components
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n, p = X.shape
        q = y.shape[1]

        # Internal centering + sample-std standardization (ddof=1) so the
        # NIPALS feature selection operates on equally-scaled columns even
        # when the upstream scaler used population std (ddof=0).
        self._x_mean = X.mean(axis=0)
        self._x_std  = X.std(axis=0, ddof=1)
        self._x_std[self._x_std < 1e-12] = 1.0
        self._y_mean = y.mean(axis=0)
        Xd = (X - self._x_mean) / self._x_std
        Yd = y - self._y_mean

        K_req = min(self.n_components, max(1, n - 1), p)
        W = np.zeros((p, K_req))
        P = np.zeros((p, K_req))
        Q = np.zeros((q, K_req))

        actual = 0
        for k in range(K_req):
            # initialize with first y-column
            u = Yd[:, [0]].copy()
            w_prev = np.zeros((p, 1))

            for _ in range(self.max_iter):
                w = Xd.T @ u
                nrm = np.linalg.norm(w)
                if nrm < 1e-12:
                    break
                w = w / nrm
                # sparsity step
                w_th = _topk_threshold(w.ravel(), self.sparsity).reshape(-1, 1)
                nrm2 = np.linalg.norm(w_th)
                if nrm2 < 1e-12:
                    w = np.zeros_like(w)
                    break
                w = w_th / nrm2

                t = Xd @ w
                tt = float(t.T @ t)
                if tt < 1e-12:
                    break
                c = (Yd.T @ t) / tt
                cc = float(c.T @ c)
                if cc < 1e-12:
                    break
                u_new = (Yd @ c) / cc

                if np.linalg.norm(u_new - u) < self.tol:
                    u = u_new
                    break
                u = u_new
                if np.allclose(w, w_prev, atol=1e-12):
                    break
                w_prev = w

            t = Xd @ w
            tt = float(t.T @ t)
            if tt < 1e-12 or np.linalg.norm(w) < 1e-12:
                # this component contributes nothing — stop
                break
            p_load = (Xd.T @ t) / tt
            q_load = (Yd.T @ t) / tt
            W[:, k] = w.ravel()
            P[:, k] = p_load.ravel()
            Q[:, k] = q_load.ravel()
            Xd = Xd - t @ p_load.T
            Yd = Yd - t @ q_load.T
            actual += 1

        self.W_ = W[:, :actual]
        self.P_ = P[:, :actual]
        self.Q_ = Q[:, :actual]
        if actual > 0:
            self.rotation_ = self.W_ @ np.linalg.pinv(self.P_.T @ self.W_)
            self.coef_ = self.rotation_ @ self.Q_.T
        else:
            self.rotation_ = np.zeros((p, 0))
            self.coef_ = np.zeros((p, q))
        self._q = q
        self._p = p
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xd = (X - self._x_mean) / self._x_std
        return Xd @ self.rotation_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xd = (X - self._x_mean) / self._x_std
        Yhat = Xd @ self.coef_ + self._y_mean
        if Yhat.shape[1] == 1:
            return Yhat.ravel()
        return Yhat
