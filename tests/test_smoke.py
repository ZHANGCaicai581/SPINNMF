import numpy as np
import scipy.sparse as sp

from spinnmf.config import SPINNMFConfig
from spinnmf.autok import fit_spin_nmf_autok


def test_smoke_autok():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(200, 300)))
    coords = rng.normal(size=(200, 2)).astype(np.float32)
    cfg = SPINNMFConfig(k_grid=(2, 3), n_seeds=3, n_jobs=1, max_iter=50)
    res = fit_spin_nmf_autok(X, coords, cfg)
    assert res.W.shape[0] == 200
    assert res.H.shape[1] == 300
    assert res.K_star in (2, 3)
