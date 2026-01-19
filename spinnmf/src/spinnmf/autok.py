from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from .config import SPINNMFConfig
from .consensus import consensus_average
from .graph import knn_graph, laplacian
from .metrics import program_redundancy, stability_across_runs
from .model import SPINNMFResult, fit_spin_nmf


@dataclass
class AutoKResult:
    K_star: int
    W: np.ndarray
    H: np.ndarray
    metrics: pd.DataFrame


def _run_one_seed(X, K, L, cfg: SPINNMFConfig, seed: int) -> SPINNMFResult:
    return fit_spin_nmf(
        X=X, K=K, L=L,
        alpha_graph=cfg.alpha_graph,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        random_state=seed,
    )


def fit_spin_nmf_autok(X, coords: np.ndarray, cfg: SPINNMFConfig) -> AutoKResult:
    A = knn_graph(coords, k=cfg.k_neighbors, metric=cfg.graph_metric)
    L = laplacian(A, normalized=True)

    records = []
    best_by_K = {}

    for K in cfg.k_grid:
        seeds = [cfg.random_state + i for i in range(cfg.n_seeds)]
        results = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_run_one_seed)(X, K, L, cfg, s) for s in seeds
        )
        W_list = [r.W for r in results]
        H_list = [r.H for r in results]

        cons = consensus_average(W_list, H_list)
        dev = float(np.mean([r.deviance for r in results]))
        stab = stability_across_runs(H_list)
        red = program_redundancy(cons.H_cons, threshold=cfg.autok_eta_red)

        records.append({"K": K, "deviance": dev, "stability": stab, "redundancy": red})
        best_by_K[K] = cons

    df = pd.DataFrame(records).sort_values("K").reset_index(drop=True)

    dev_min = df["deviance"].min()
    df["near_opt"] = df["deviance"] <= (1.0 + cfg.autok_delta) * dev_min
    candidates = df[
        (df["near_opt"]) &
        (df["stability"] >= cfg.autok_tau_stab) &
        (df["redundancy"] <= cfg.autok_eta_red)
    ]

    if len(candidates) == 0:
        K_star = int(df.loc[df["deviance"].idxmin(), "K"])
    else:
        K_star = int(candidates.sort_values(["K"]).iloc[0]["K"])

    cons_star = best_by_K[K_star]
    return AutoKResult(K_star=K_star, W=cons_star.W_cons, H=cons_star.H_cons, metrics=df)
