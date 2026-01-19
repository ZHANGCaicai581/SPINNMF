from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass(frozen=True)
class SPINNMFConfig:
    # graph
    k_neighbors: int = 15
    graph_metric: str = "euclidean"

    # model
    max_iter: int = 300
    tol: float = 1e-5
    alpha_graph: float = 0.2  # graph smoothing strength
    random_state: int = 0

    # consensus
    n_seeds: int = 10
    n_jobs: int = 1

    # autok
    k_grid: Sequence[int] = (2, 3, 4, 5, 6, 7, 8)
    autok_delta: float = 0.02      # near-optimal deviance margin
    autok_tau_stab: float = 0.75   # min stability
    autok_eta_red: float = 0.90    # max redundancy
    holdout_frac: float = 0.05     # for simple heldout fit proxy (optional)

    # gene selection
    gene_mode: str = "hvg"         # hvg / svg / all
    n_genes: int = 2000
