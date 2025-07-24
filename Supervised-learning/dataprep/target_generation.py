"""target_generation.py

Label-creation utilities for the ML-trading project.

Goals
-----
* **Flexible horizon** (N trading days).
* **Binary targets**: price exceeds ±X % threshold by horizon end.
* **Regression targets**: raw N-day log return.
* **Class prevalence diagnostics** per date and per ticker.
* Stored with the same `(Datetime, Ticker)` index as the feature set for
  seamless `join` operations in downstream loaders.

Assumptions
-----------
* Close prices come from the **feature Parquet** produced by
  `data_pipeline.py`. No additional download call is made here—keeps the two
  pipelines modular.
* The DataFrame index order is `(Datetime, Ticker)` - identical to the feature
  file.
* The input Parquet must contain the `Close` column; if not, raise an error.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def n_day_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Compute forward log return over *horizon* periods (trading-day aligned)."""
    fwd = close.groupby(level=1).shift(-horizon)
    return np.log(fwd) - np.log(close)


def binary_threshold(close: pd.Series, pct: float, horizon: int, direction: str = "up") -> pd.Series:
    """Label 1 if price moves **above** (+) or **below** (-) given *pct* in *horizon* days."""
    assert direction in {"up", "down"}
    ret = n_day_log_return(close, horizon)
    thresh = np.log1p(pct) if direction == "up" else -np.log1p(pct)
    return (ret > thresh).astype(int) if direction == "up" else (ret < thresh).astype(int)


def prevalence_table(binary_target: pd.Series) -> pd.DataFrame:
    """Return prevalence (%) by year and overall."""
    df = binary_target.to_frame("y")
    df["year"] = df.index.get_level_values(0).year
    tbl = df.groupby("year")["y"].mean().to_frame("positive_rate")
    tbl.loc["ALL", "positive_rate"] = df["y"].mean()
    return tbl.round(4)


def build_and_save_labels(
    feature_path: Path,
    out_path: Path,
    horizon: int,
    pct: float | None = None,
    direction: str = "up",
) -> tuple[Path, pd.DataFrame]:
    """Generate labels and save to Parquet.

    Parameters
    ----------
    feature_path : Path
        Parquet file produced by data_pipeline containing at least a *Close*
        column.
    out_path : Path
        Destination Parquet file for labels.
    horizon : int
        Forward window in trading days.
    pct : float | None
        If provided, create a binary threshold label (e.g. 0.05 for ±5 %).
        If None, store raw log return as regression target.
    direction : str
        "up" (default) for upward moves; "down" for crashes. Ignored if pct is
        None.
    """
    df = pd.read_parquet(feature_path, engine="pyarrow")

    if 'ticker_type' in df:
        df = df[df['ticker_type'] == 'asset'] 

    if "Close" not in df.columns:
        raise ValueError("Feature parquet must contain a 'Close' column for target generation.")

    close = df["Close"]

    if pct is None:
        label = n_day_log_return(close, horizon)
    else:
        label = binary_threshold(close, pct, horizon, direction)

    # Align index and drop rows with NaN (no future price)
    label = label.dropna().rename("target")

    if isinstance(label, pd.Series):
        label = pd.DataFrame(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    label.to_parquet(out_path, engine="pyarrow", compression="snappy")
    return out_path, label


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate N‑day targets from feature set")
    p.add_argument("feature_path", type=Path, help="Parquet produced by data_pipeline.py")
    p.add_argument("out_path", type=Path, help="Output Parquet for labels")
    p.add_argument("--horizon", type=int, default=7, help="Forward horizon in trading days")
    p.add_argument("--pct", type=float, default=None, help="Percent move (e.g. 0.05). If omitted, create regression target")
    p.add_argument("--direction", choices=["up", "down"], default="up", help="Direction for binary label")
    args = p.parse_args()

    path, lbl = build_and_save_labels(
        feature_path=args.feature_path,
        out_path=args.out_path,
        horizon=args.horizon,
        pct=args.pct,
        direction=args.direction,
    )
    print(f"Saved labels in: {path} | horizon={args.horizon} | pct={args.pct} | count={lbl.shape[0]}")
