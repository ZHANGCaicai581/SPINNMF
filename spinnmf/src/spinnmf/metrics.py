from __future__ import annotations

import numpy as np


def program_redundancy(H: np.ndarray, threshold: float = 0.9) -> float:
    eps = 1e-8
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)
    sim = Hn @ Hn.T
    np.fill_diagonal(sim, -np.inf)
    max_sim = np.max(sim, axis=1)
    return float(np.mean(max_sim))


def stability_across_runs(H_list) -> float:
    eps = 1e-8
    sims = []
    for i in range(len(H_list)):
        Hi = H_list[i] / (np.linalg.norm(H_list[i], axis=1, keepdims=True) + eps)
        for j in range(i + 1, len(H_list)):
            Hj = H_list[j] / (np.linalg.norm(H_list[j], axis=1, keepdims=True) + eps)
            sims.append(float(np.mean(np.max(Hi @ Hj.T, axis=1))))
    return float(np.mean(sims)) if sims else 1.0
