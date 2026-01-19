#!/usr/bin/env python3
"""
Meta-analysis and publication-style figures for multiple heart SPINNMF runs.

Inputs: assumes each run already has analysis_report/ from your previous script.
Also consumes merged_summary/cross_sample_factor_matching.csv if present.

Outputs (default under ./heart_meta_analysis/results/):
  - nmi_ari_per_run.png
  - domain_purity_boxplot.png
  - top_domain_population.csv (dominant population per domain)
  - top_boundary_edges_agg.csv + boundary_edges_bar.png
  - factor_match_corr_hist.png (if merged_summary exists)
  - summary.txt (key numbers)
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- user config ----------------
RUN_DIRS = [
    Path("/Users/caicaizhang/Library/CloudStorage/Dropbox/Spatial_NMF_test/heart_20251110_180023"),  # R77_4C4
    Path("/Users/caicaizhang/Library/CloudStorage/Dropbox/Spatial_NMF_test/heart_20251114_131458"),  # R78_4C15
    Path("/Users/caicaizhang/Library/CloudStorage/Dropbox/Spatial_NMF_test/heart_20251110_163555"),  # R78_4C12
]
RUN_ORDER = [p.name for p in RUN_DIRS]

# Custom pretty labels for x-axis
RUN_LABELS = {
    "heart_20251110_180023": "R77_4C4",
    "heart_20251114_131458": "R78_4C15",
    "heart_20251110_163555": "R78_4C12",
}

OUTDIR = Path(__file__).resolve().parent / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def load_run_metrics(run_dir: Path) -> dict:
    rpt = run_dir / "analysis_report"
    if not rpt.exists():
        raise FileNotFoundError(f"analysis_report not found for {run_dir}")
    fn = {
        "nmi_ari": rpt / "domain_vs_population_scores.csv",
        "purity": rpt / "domain_purity.csv",
        "domain_sizes": rpt / "domain_sizes.csv",
        "domain_pop_frac": rpt / "domain_x_population_row_fraction.csv",
        "top_edges": rpt / "top_boundary_edges.csv",
        "summary": rpt / "analysis_summary.json",
    }
    for k, p in fn.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k}: {p}")
    obj = {
        "run_dir": run_dir,
        "nmi_ari": pd.read_csv(fn["nmi_ari"]),
        "purity": pd.read_csv(fn["purity"], index_col=0).squeeze("columns"),
        "domain_sizes": pd.read_csv(fn["domain_sizes"], index_col=0).squeeze("columns"),
        "domain_pop_frac": pd.read_csv(fn["domain_pop_frac"], index_col=0),
        "top_edges": pd.read_csv(fn["top_edges"]),
        "summary": json.loads(fn["summary"].read_text()),
    }
    return obj


def load_factor_matching(base_dir: Path):
    merged = base_dir / "merged_summary"
    f_corr = merged / "cross_sample_factor_matching.csv"
    f_shared = merged / "cross_sample_shared_factors.csv"
    if f_corr.exists():
        df_corr = pd.read_csv(f_corr)
    else:
        df_corr = None
    if f_shared.exists():
        df_shared = pd.read_csv(f_shared)
    else:
        df_shared = None
    return df_corr, df_shared


def plot_nmi_ari(objs, out_png: Path):
    rows = []
    for o in objs:
        row = o["nmi_ari"].iloc[0].to_dict()
        row["run"] = o["run_dir"].name
        rows.append(row)
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6, 4), dpi=180)
    x = np.arange(len(df))
    w = 0.35
    plt.bar(x - w/2, df["NMI(domain,pop)"], width=w, label="NMI")
    plt.bar(x + w/2, df["ARI(domain,pop)"], width=w, label="ARI")
    plt.xticks(x, df["run"], rotation=20, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Domain vs population agreement")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_purity(objs, out_png: Path):
    rows = []
    for o in objs:
        for d, val in o["purity"].items():
            rows.append({"run": o["run_dir"].name, "domain": d, "purity": val})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6, 4), dpi=180)
    import seaborn as sns
    palette = sns.color_palette("colorblind", n_colors=len(RUN_ORDER))
    ax = sns.boxplot(data=df, x="run", y="purity", order=RUN_ORDER, palette=palette, width=0.6)
    sns.stripplot(data=df, x="run", y="purity", order=RUN_ORDER, color="k", size=3, alpha=0.55)
    labels = [RUN_LABELS.get(r, r) for r in RUN_ORDER]
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    plt.ylim(0, 1.05)
    plt.title("Domain purity", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    return df


def summarize_domain_mapping(objs, out_csv: Path):
    rows = []
    for o in objs:
        frac = o["domain_pop_frac"]
        for dom in frac.index:
            row = frac.loc[dom]
            top_pop = row.idxmax()
            purity = float(row.max())
            rows.append({
                "run": o["run_dir"].name,
                "domain": dom,
                "top_population": top_pop,
                "purity": purity,
            })
    pd.DataFrame(rows).sort_values(["run", "domain"]).to_csv(out_csv, index=False)


def aggregate_top_edges(objs, out_csv: Path, out_png: Path, top_n: int = 20):
    frames = []
    for o in objs:
        df = o["top_edges"].copy()
        df["run"] = o["run_dir"].name
        # normalize column names across versions
        if "domain_a" in df.columns and "domain_b" in df.columns:
            df = df.rename(columns={"domain_a": "dom_a", "domain_b": "dom_b"})
        frames.append(df)
    all_edges = pd.concat(frames, ignore_index=True)
    agg = (
        all_edges
        .groupby(["dom_a", "dom_b"], as_index=False)["edge_count"]
        .sum()
        .sort_values("edge_count", ascending=False)
    )
    agg.head(top_n).to_csv(out_csv, index=False)

    plt.figure(figsize=(6, 4.5), dpi=180)
    head = agg.head(top_n)
    plt.barh(np.arange(len(head)), head["edge_count"], color="slategray")
    labels = [f"{a}-{b}" for a, b in zip(head["dom_a"], head["dom_b"])]
    plt.yticks(np.arange(len(head)), labels)
    plt.xlabel("Edge count (summed across runs)")
    plt.title("Top domain boundaries")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_factor_corr(df_corr: pd.DataFrame, out_png: Path):
    plt.figure(figsize=(6, 4), dpi=180)
    plt.hist(df_corr["corr"], bins=20, color="steelblue", alpha=0.8)
    plt.xlabel("Cross-sample factor correlation")
    plt.ylabel("Count")
    plt.title("Factor alignment across runs")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def write_summary(objs, purity_df: pd.DataFrame, df_corr, out_txt: Path):
    lines = []
    lines.append("Heart SPINNMF meta-analysis\n")
    lines.append("Runs:\n")
    for o in objs:
        lines.append(f"  - {o['run_dir'].name}")
    lines.append("")

    # NMI / ARI
    lines.append("NMI / ARI by run:")
    for o in objs:
        row = o["nmi_ari"].iloc[0]
        lines.append(f"  {o['run_dir'].name}: NMI={row['NMI(domain,pop)']:.3f}, ARI={row['ARI(domain,pop)']:.3f}")
    lines.append("")

    # Purity
    lines.append("Domain purity (median per run):")
    med = purity_df.groupby("run")["purity"].median()
    for run, val in med.items():
        lines.append(f"  {run}: median purity={val:.3f}")
    lines.append("")

    if df_corr is not None:
        lines.append(f"Cross-sample factor pairs: n={len(df_corr)}, mean corr={df_corr['corr'].mean():.3f}")
        gt = (df_corr["corr"] >= 0.5).mean()
        lines.append(f"  Fraction corr>=0.5: {gt:.2%}")
    else:
        lines.append("Cross-sample factor matching: not available (missing merged_summary).")
    lines.append("")

    out_txt.write_text("\n".join(lines))


def main():
    objs = [load_run_metrics(rd) for rd in RUN_DIRS]

    plot_nmi_ari(objs, OUTDIR / "nmi_ari_per_run.png")
    purity_df = plot_purity(objs, OUTDIR / "domain_purity_boxplot.png")
    summarize_domain_mapping(objs, OUTDIR / "top_domain_population.csv")
    aggregate_top_edges(objs, OUTDIR / "top_boundary_edges_agg.csv", OUTDIR / "boundary_edges_bar.png")

    # cross-sample factor matching (optional)
    df_corr, df_shared = load_factor_matching(RUN_DIRS[0].parent)
    if df_corr is not None:
        plot_factor_corr(df_corr, OUTDIR / "factor_match_corr_hist.png")

    write_summary(objs, purity_df, df_corr, OUTDIR / "summary.txt")
    print(f"✅ Meta-analysis done. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
