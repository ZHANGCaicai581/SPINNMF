from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def save_metrics(df: pd.DataFrame, outdir: Path, name: str = "autok_metrics.csv"):
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / name, index=False)


def save_summary(outdir: Path, summary: dict[str, any], name: str = "summary.json"):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / name).write_text(json.dumps(summary, indent=2))
