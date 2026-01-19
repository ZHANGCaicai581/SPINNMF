from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc

from .autok import fit_spin_nmf_autok
from .config import SPINNMFConfig
from .io import get_spatial_coords
from .report import save_metrics, save_summary


def main():
    parser = argparse.ArgumentParser("spinnmf")
    parser.add_argument("--h5ad", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--k_grid", default="2,3,4,5,6,7,8")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--alpha_graph", type=float, default=0.2)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    coords = get_spatial_coords(adata)
    X = adata.X

    cfg = SPINNMFConfig(
        k_grid=tuple(int(x.strip()) for x in args.k_grid.split(",") if x.strip()),
        n_seeds=args.n_seeds,
        k_neighbors=args.k_neighbors,
        alpha_graph=args.alpha_graph,
        n_jobs=args.n_jobs,
    )

    res = fit_spin_nmf_autok(X, coords, cfg)

    np.save(outdir / "W.npy", res.W)
    np.save(outdir / "H.npy", res.H)
    save_metrics(res.metrics, outdir)

    summary = {"K_star": res.K_star, "config": json.loads(json.dumps(cfg.__dict__))}
    save_summary(outdir, summary)

    print(f"[SPINNMF] Done. Selected K*={res.K_star}. Outputs: {outdir}")


if __name__ == "__main__":
    main()
