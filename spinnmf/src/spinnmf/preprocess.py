from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _ensure_csr(X):
    if sp.issparse(X):
        return X.tocsr(copy=True)
    return sp.csr_matrix(X)


def select_hvg_indices(X, n_top=2000):
    X = _ensure_csr(X)
    mu = np.asarray(X.mean(axis=0)).ravel()
    sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    var = np.maximum(sq - mu ** 2, 0.0)
    fano = var / (mu + 1e-8)
    n_top = min(n_top, X.shape[1])
    idx = np.argsort(fano)[-n_top:]
    return np.sort(idx)


def select_svg_indices(X, coords, n_top=2000, k=12, chunk=512):
    if coords is None:
        raise ValueError("Spatial coordinates required for SVG selection.")
    X = _ensure_csr(X)
    n = coords.shape[0]
    k = min(k, max(1, n - 1))
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(coords.astype(np.float32))
    dist, ind = nbrs.kneighbors(coords.astype(np.float32))
    rows = np.repeat(np.arange(n), k)
    cols = ind[:, 1:].ravel()
    data = np.ones_like(cols, dtype=np.float32)
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    W = (W + W.T).astype(np.float32)
    rs = np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1e-8)
    WR = sp.diags(1.0 / rs, dtype=np.float32) @ W
    J = X.shape[1]
    scores = np.zeros(J, dtype=np.float32)
    for start in range(0, J, chunk):
        end = min(J, start + chunk)
        Xb = X[:, start:end].toarray().astype(np.float32)
        Xb = Xb - Xb.mean(axis=0, keepdims=True)
        Yb = WR @ Xb
        num = np.sum(Xb * Yb, axis=0)
        den = np.sum(Xb * Xb, axis=0) + 1e-12
        scores[start:end] = num / den
    n_top = min(n_top, J)
    idx = np.argsort(scores)[-n_top:]
    return np.sort(idx)


def select_genes(
    X,
    coords,
    mode: str = "hvg",
    n_genes: int = 2000,
    svg_k: int = 12,
    svg_chunk: int = 512,
):
    if mode == "all":
        mask = np.ones(X.shape[1], dtype=bool)
    elif mode == "hvg":
        idx = select_hvg_indices(X, n_top=n_genes)
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[idx] = True
    elif mode == "svg":
        idx = select_svg_indices(X, coords, n_top=n_genes, k=svg_k, chunk=svg_chunk)
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[idx] = True
    else:
        raise ValueError("mode must be one of: 'all', 'hvg', 'svg'")
    return mask
