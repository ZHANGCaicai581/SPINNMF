from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF


@dataclass
class SPINNMFResult:
    W: np.ndarray           # (N, K)
    H: np.ndarray           # (K, G)
    deviance: float
    n_iter: int
    converged: bool


def _to_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def _poisson_deviance(X, WH, eps=1e-10) -> float:
    X_d = X
    WH_d = np.maximum(WH, eps)
    if sp.issparse(X_d):
        X_coo = X_d.tocoo()
        log_ratio = np.log(np.maximum(X_coo.data, eps)) - np.log(WH_d[X_coo.row, X_coo.col])
        term = X_coo.data * log_ratio
        return float(2.0 * (term.sum() - (X_d.sum() - WH_d.sum())))
    term = X_d * (np.log(np.maximum(X_d, eps)) - np.log(WH_d))
    return float(2.0 * (term.sum() - (X_d.sum() - WH_d.sum())))


def fit_spin_nmf(
    X,
    K: int,
    L: Optional[sp.csr_matrix] = None,
    alpha_graph: float = 0.2,
    max_iter: int = 300,
    tol: float = 1e-5,
    random_state: int = 0,
) -> SPINNMFResult:
    X_csr = _to_csr(X)
    nmf = NMF(
        n_components=K,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    W = nmf.fit_transform(X_csr)
    H = nmf.components_.copy()

    if L is not None and alpha_graph > 0:
        L = L.tocsr().astype(np.float32)
        step = min(0.25, 1.0 / (np.max(np.abs(L.diagonal())) + 1e-6))
        for _ in range(20):
            LW = L @ W
            W_new = np.maximum(W - alpha_graph * step * LW, 0.0)
            if np.linalg.norm(W_new - W) / (np.linalg.norm(W) + 1e-8) < 1e-4:
                W = W_new
                break
            W = W_new

    WH = W @ H
    dev = _poisson_deviance(X_csr, WH)
    return SPINNMFResult(W=W, H=H, deviance=dev, n_iter=nmf.n_iter_, converged=(nmf.n_iter_ < max_iter))
