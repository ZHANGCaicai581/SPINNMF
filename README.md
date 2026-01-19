# SPINNMF

![SPINNMF workflow](docs/method_workflow.png)

## 安装

-（发布到 PyPI 后）`pip install spinnmf`
- 开发模式：在源码根目录运行 `pip install -e .[dev]`

## 快速使用

命令行运行 AutoK：
```bash
spinnmf \
  --h5ad /path/to/spatial.h5ad \
  --outdir /path/to/outdir \
  --k_grid 2,3,4,5 \
  --n_seeds 5 \
  --n_jobs 4
```

Python 调用：
```python
import anndata as ad
from spinnmf import fit_spin_nmf_autok, config

adata = ad.read_h5ad("path/to/spatial.h5ad")
cfg = config.SPINNMFConfig(k_grid=(2, 3, 4), n_seeds=3, n_jobs=1)
res = fit_spin_nmf_autok(adata.X, adata.obsm["spatial"], cfg)
print(res.K_star, res.W.shape, res.H.shape)
```

## 测试与构建

```bash
pytest
python -m build
twine check dist/*
```