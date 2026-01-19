from __future__ import annotations
from typing import Optional, Tuple

import anndata as ad
import numpy as np


def get_spatial_coords(
    adata: ad.AnnData,
    basis_priority: Tuple[str, ...] = ("spatial", "X_spatial", "spatial_global", "X_CCF"),
) -> np.ndarray:
    for key in basis_priority:
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key])
            if coords.ndim == 2 and coords.shape[1] >= 2:
                return coords[:, :2].astype(np.float32)
    raise ValueError(f"No spatial coordinates found in adata.obsm among: {basis_priority}")


def get_counts_matrix(adata: ad.AnnData, layer: Optional[str] = None):
    if layer and layer in adata.layers:
        return adata.layers[layer]
    return adata.X


def subset_genes(adata: ad.AnnData, gene_mask) -> ad.AnnData:
    return adata[:, gene_mask].copy()
