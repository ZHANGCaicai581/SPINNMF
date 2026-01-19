#!/usr/bin/env python3
"""
Regenerate spatial plots for a single run with titles showing sample ID.

Outputs (saved under run_dir/post_analysis):
  - spatial__populations__K9_title.png
  - spatial__domain__K9_title.png  (domain legend ordered by dominant factor)
  - spatial__factor_dominant__K9_title.png (single map colored by dominant factor)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- user config ----------------
RUN_DIR = Path("/Users/caicaizhang/Library/CloudStorage/Dropbox/Spatial_NMF_test/heart_20251110_163555")
SAMPLE_ID = "R78_4C12"
MODEL_BASENAME = "selected_model_K9_with_domain.h5ad"

# plot settings
FIGSIZE = (6.4, 6.4)
DPI = 300
POINT_SIZE = 2.0
ALPHA = 0.95
RIGHT_MARGIN = 0.82
LEGEND_FONT = 12
LEGEND_MARKERSIZE = 10


def build_palette(categories):
    """Return a dict category -> color, expanding across a few Matplotlib palettes."""
    cats = list(categories)
    palettes = ["Set2", "Set3", "Accent", "Dark2", "Paired", "tab20", "tab20b", "tab20c"]
    colors = []
    for name in palettes:
        cmap = plt.get_cmap(name)
        n = getattr(cmap, "N", 20)
        take = max(0, min(len(cats) - len(colors), n))
        colors += [tuple(cmap(i)) for i in range(take)]
        if len(colors) >= len(cats):
            break
    if len(colors) < len(cats):
        gist = plt.get_cmap("gist_ncar")
        colors += [tuple(gist(v)) for v in np.linspace(0, 1, len(cats) - len(colors), endpoint=False)]
    return {c: colors[i] for i, c in enumerate(cats)}


def rotate_180_center(xy: np.ndarray) -> np.ndarray:
    c = xy.mean(axis=0, keepdims=True)
    return (2 * c) - xy


def factor_dominance(adata) -> pd.DataFrame:
    """Map domain -> dominant factor fraction for ordering legends."""
    dom = adata.obs["domain"].astype(str)
    F = adata.obsm["fnmf_F"]
    rows = []
    for d in sorted(dom.unique()):
        idx = np.where(dom.values == d)[0]
        top = F[idx].argmax(1)
        vc = pd.Series(top).value_counts(normalize=True)
        rows.append({"domain": d, "best_factor": int(vc.idxmax()), "best_frac": float(vc.max()), "n": len(idx)})
    return pd.DataFrame(rows)


def plot_population(adata, coords_rot, out_png: Path):
    pops = adata.obs["populations"].astype(str)
    uniq = [u for u in pd.unique(pops) if u not in ("nan", "NA", "None")]
    pal = build_palette(uniq)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = plt.gca()
    plt.subplots_adjust(right=RIGHT_MARGIN)
    for u in uniq:
        m = pops == u
        ax.scatter(coords_rot[m, 0], coords_rot[m, 1], s=POINT_SIZE, alpha=ALPHA, c=[pal[u]], edgecolors="none")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=LEGEND_MARKERSIZE,
               markerfacecolor=pal[u], markeredgecolor="none", label=str(u))
        for u in uniq
    ]
    leg = ax.legend(
        handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
        ncol=1, prop={"size": LEGEND_FONT, "weight": "bold"},
        handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0,
    )
    for t in leg.get_texts():
        t.set_fontweight("bold")
    ax.set_title(f"{SAMPLE_ID} — populations", fontsize=13, fontweight="bold")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_domain(adata, coords_rot, dom_order, out_png: Path):
    dom = pd.Categorical(adata.obs["domain"].astype(str), categories=dom_order, ordered=True)
    uniq = dom_order
    pal = build_palette(uniq)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = plt.gca()
    plt.subplots_adjust(right=RIGHT_MARGIN)
    vals = dom.astype(str)
    for u in uniq:
        m = vals == u
        ax.scatter(coords_rot[m, 0], coords_rot[m, 1], s=POINT_SIZE, alpha=ALPHA, c=[pal[u]], edgecolors="none")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=LEGEND_MARKERSIZE,
               markerfacecolor=pal[u], markeredgecolor="none", label=str(u))
        for u in uniq
    ]
    leg = ax.legend(
        handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
        ncol=1, prop={"size": LEGEND_FONT, "weight": "bold"},
        handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0,
    )
    for t in leg.get_texts():
        t.set_fontweight("bold")
    ax.set_title(f"{SAMPLE_ID} — domains (ordered by factor)", fontsize=13, fontweight="bold")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_factor_dominant(adata, coords_rot, out_png: Path):
    F = adata.obsm["fnmf_F"]
    fac_max = F.argmax(1)
    K = F.shape[1]
    colors = plt.get_cmap("tab20").colors
    pal = {k: colors[k % len(colors)] for k in range(K)}

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = plt.gca()
    plt.subplots_adjust(right=RIGHT_MARGIN)
    for k in range(K):
        m = fac_max == k
        if m.sum() == 0:
            continue
        ax.scatter(coords_rot[m, 0], coords_rot[m, 1], s=POINT_SIZE, alpha=ALPHA, c=[pal[k]], edgecolors="none")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=LEGEND_MARKERSIZE,
               markerfacecolor=pal[k], markeredgecolor="none", label=f"F{k}")
        for k in range(K)
    ]
    leg = ax.legend(
        handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
        ncol=1, prop={"size": LEGEND_FONT, "weight": "bold"},
        handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0,
    )
    for t in leg.get_texts():
        t.set_fontweight("bold")
    ax.set_title(f"{SAMPLE_ID} — dominant factor per spot", fontsize=13, fontweight="bold")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    adata = sc.read_h5ad(RUN_DIR / "selected" / MODEL_BASENAME)
    coords_rot = rotate_180_center(np.asarray(adata.obsm["spatial"], float))
    outdir = RUN_DIR / "post_analysis"
    outdir.mkdir(exist_ok=True)

    # domain ordering by dominant factor
    map_df = factor_dominance(adata)
    dom_order = map_df.sort_values(["best_factor", "best_frac"], ascending=[True, False])["domain"].tolist()

    plot_population(adata, coords_rot, outdir / "spatial__populations__K9_title.png")
    plot_domain(adata, coords_rot, dom_order, outdir / "spatial__domain__K9_title.png")
    plot_factor_dominant(adata, coords_rot, outdir / "spatial__factor_dominant__K9_title.png")
    print("Saved to", outdir)


if __name__ == "__main__":
    main()
