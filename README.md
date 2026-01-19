# SPINNMF

![SPINNMF workflow](docs/method_workflow.png)

Stable spatial pattern discovery via AutoK consensus graph-regularized Poisson NMF.

## Overview
- Poisson/KL NMF on counts with spatial graph regularization.
- Multi-start consensus alignment for reproducible factors.
- AutoK chooses K using fit–stability–redundancy diagnostics.
- Optional domain segmentation from continuous program activities.

## Requirements & install
- Python >=3.9
- PyPI: `pip install spinnmf`
- Dev (repo root): `pip install -e .[dev]`

## Inputs
- `h5ad` with counts in `adata.X` (nonnegative) and coordinates in `adata.obsm["spatial"]`.
- Graph is built internally from coordinates (kNN).

## Quick start (CLI)
```bash
spinnmf \
  --h5ad /path/to/spatial.h5ad \
  --outdir /path/to/outdir \
  --k_grid 2,3,4,5 \
  --n_seeds 5 \
  --n_jobs 4
```
Key flags:
- `--k_grid` comma-separated candidate Ks
- `--n_seeds` consensus runs per K
- `--alpha` spatial smoothness (default 1e-2)
- `--knn` neighbors for spatial graph (default 6 for Visium-like)

## Quick start (Python)
```python
import anndata as ad
from spinnmf import fit_spin_nmf_autok, config

adata = ad.read_h5ad("path/to/spatial.h5ad")
coords = adata.obsm["spatial"]
cfg = config.SPINNMFConfig(k_grid=(2, 3, 4), n_seeds=3, n_jobs=1)
res = fit_spin_nmf_autok(adata.X, coords, cfg)

print("K*", res.K_star)
print("W shape", res.W.shape, "H shape", res.H.shape)
```

## Outputs
- `W.npy`, `H.npy`: consensus spatial activities and gene loadings.
- `autok_metrics.csv`: fit / stability / redundancy per K.
- `summary.json`: selected `K*` and run configuration.
- Optional domain labels (if downstream clustering on W is requested).

## Notes on AutoK
- Defaults: stability threshold ~0.8, redundancy threshold ~0.9, parsimonious deviance tolerance 0.02.
- Provide a sensible `k_grid` (e.g., 2–8 for small panels; broader for larger panels).

## Development / tests
```bash
pip install -e .[dev]
pytest
```

## Build & release
```bash
python -m build
twine check dist/*
# twine upload dist/*   # requires PyPI token
```
GitHub Actions workflow (`.github/workflows/ci.yml`) runs tests on PRs/pushes and, on tags `v*`, builds and publishes to PyPI using `PYPI_API_TOKEN` secret.

## License
MIT

## Citation
If you use SPINNMF, please cite the accompanying manuscript (title placeholder). A BibTeX entry will be provided upon publication.