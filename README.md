# SPINNMF

![SPINNMF workflow](docs/method_workflow.png)

Stable spatial pattern discovery via AutoK consensus graph-regularized Poisson NMF.

## Install

- PyPI: `pip install spinnmf`
- Dev (repo root): `pip install -e .[dev]`

Requires Python >=3.9.

## Quick start

CLI (AutoK):
```bash
spinnmf \
  --h5ad /path/to/spatial.h5ad \
  --outdir /path/to/outdir \
  --k_grid 2,3,4,5 \
  --n_seeds 5 \
  --n_jobs 4
```

Python API:
```python
import anndata as ad
from spinnmf import fit_spin_nmf_autok, config

adata = ad.read_h5ad("path/to/spatial.h5ad")
cfg = config.SPINNMFConfig(k_grid=(2, 3, 4), n_seeds=3, n_jobs=1)
res = fit_spin_nmf_autok(adata.X, adata.obsm["spatial"], cfg)
print(res.K_star, res.W.shape, res.H.shape)
```

Outputs (CLI):
- `W.npy`, `H.npy`: consensus factors
- `autok_metrics.csv`: stability / redundancy per K
- `summary.json`: selected K* and config

## Test and build

```bash
pytest
python -m build
twine check dist/*
```