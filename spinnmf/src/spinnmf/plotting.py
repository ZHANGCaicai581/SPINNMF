from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_curve(df: pd.DataFrame, y: str, outpath: Path):
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(df["K"], df[y], marker="o")
    plt.xlabel("K")
    plt.ylabel(y)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
