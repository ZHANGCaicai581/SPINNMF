from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class ConsensusResult:
    W_cons: np.ndarray
    H_cons: np.ndarray
    perm_list: List[np.ndarray]


def _corr_sim(A: np.ndarray, B: np.ndarray, eps=1e-8) -> np.ndarray:
    A2 = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B2 = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A2 @ B2.T


def align_to_reference(H_ref: np.ndarray, H: np.ndarray) -> np.ndarray:
    sim = _corr_sim(H_ref, H)
    cost = -sim
    r, c = linear_sum_assignment(cost)
    perm = c[np.argsort(r)]
    return perm


def consensus_average(W_list: List[np.ndarray], H_list: List[np.ndarray]) -> ConsensusResult:
    H_ref = H_list[0]
    perms = [np.arange(H_ref.shape[0])]
    W_aligned = [W_list[0]]
    H_aligned = [H_ref]

    for i in range(1, len(H_list)):
        perm = align_to_reference(H_ref, H_list[i])
        perms.append(perm)
        W_aligned.append(W_list[i][:, perm])
        H_aligned.append(H_list[i][perm, :])

    W_cons = np.mean(np.stack(W_aligned, axis=0), axis=0)
    H_cons = np.mean(np.stack(H_aligned, axis=0), axis=0)
    return ConsensusResult(W_cons=W_cons, H_cons=H_cons, perm_list=perms)
