from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def knn_graph(coords: np.ndarray, k: int = 15, metric: str = "euclidean") -> sp.csr_matrix:
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric)
    nn.fit(coords)
    dists, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols = idx[:, 1:].reshape(-1)
    data = np.ones_like(cols, dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    return A


def laplacian(A: sp.csr_matrix, normalized: bool = True) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    if normalized:
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        D_inv_sqrt = sp.diags(deg_inv_sqrt.astype(np.float32))
        L = sp.eye(A.shape[0], dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = sp.diags(deg.astype(np.float32))
        L = D - A
    return L.tocsr()
