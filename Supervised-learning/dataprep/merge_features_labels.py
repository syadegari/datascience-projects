"""merge_features_labels.py

Step 1: Merge feature and label Parquet files into a single model matrix.

Responsibilities
----------------
- Load feature matrix Parquet and label Parquet.
- Perform an outer join on (Datetime, Ticker).
- Assert that no unexpected NaNs remain in core features or labels.
- Write out a unified `model_matrix.parquet` for downstream model code.

Usage
-----
python merge_features_labels.py \
    --features data/features_1d_2010-01-01_today.parquet \
    --labels data/labels_7d_5pct.parquet \
    --output data/model_matrix.parquet
"""
import sys
from pathlib import Path
import argparse

import pandas as pd


PREDICTIVE = ["log_ret", "z_log_ret", "vol_scaled"]


def apply_nan_policy(df: pd.DataFrame, feature_cols: list[str], fill_value: float = 0.0) -> pd.DataFrame:
    """
    Apply a consistent NaN-handling policy to the **model matrix** produced by
    `merge_features_labels.py`.

    How it is done
    --------------
    1. **Predictive feature columns** that can be missing (currently just
    `vol_scaled`) are filled with `0.0` **and** accompanied by a binary mask
    column `<feature>_mask` where `1` denotes "was NaN in the raw data".
    2. **Label column** `target` is left untouched; auxiliary rows remain NaN.
    3. Non-numeric columns (e.g. `ticker_type`) are left unchanged.

    Why 0.0 for `fill_value`?
    -------------------------
    log_ret and z_log_ret are already ~ N(0, 1); a zero fill is truly “average”.
    The mask bit lets the net still exploit the information that the value was
    missing (e.g. vol_scaled for macro tickers).
    
    Why this is safe
    ----------------
    Assets keep fully observed targets.
    Auxiliary rows still carry NaNs in target, but they’re never selected as
    (X,y) pairs because the Dataset filters ticker_type == "asset".

    DL frameworks love it.
    All numeric tensors can go through LayerNorm/softmax without blowing up
    (NaN would do that). The mask bits are ordinary inputs a linear layer can
    ingest or you can split them into a Boolean mask for attention later.

    Tree baselines unaffected.
    If you pipe the filled DataFrame into LightGBM, it will treat 0 as a valid
    numeric value; you can include the mask columns or drop them.
    """
    for col in feature_cols:
        if col not in df.columns:
            print(f"WARNING: column '{col}' not in DataFrame - skipping", file=sys.stderr)
            continue
        mask_col = f"{col}_mask"
        df[mask_col] = df[col].isna().astype("int8")
        df.fillna({col: fill_value}, inplace=True)
    return df


def merge_and_validate(features_path: Path, labels_path: Path) -> pd.DataFrame:
    # Load inputs
    feat: pd.DataFrame = pd.read_parquet(features_path, engine="pyarrow")
    lbl: pd.DataFrame = pd.read_parquet(labels_path, engine="pyarrow")

    assert isinstance(feat, pd.DataFrame) and 'ticker_type' in feat
    assert isinstance(lbl, pd.DataFrame) and lbl.shape[1] == 1

    # Merge on (Datetime, Ticker) and assign mask
    df = feat.join(lbl, how="outer")
    apply_nan_policy(df, PREDICTIVE)


    assets_mask = df["ticker_type"] == "asset"
    # 1. Asset rows must have labels
    missing_target_assets = df.loc[assets_mask, "target"].isna().sum()
    if missing_target_assets > 0:
        raise ValueError(f"Found {missing_target_assets} asset rows with no target label")

    # 2. Predictive features should not be NaN for asset rows (first lag row already dropped in pipeline)
    predictive_cols = [c for c in ["log_ret", "z_log_ret"] if c in df.columns]
    if df.loc[assets_mask, predictive_cols].isna().any().any():
        raise ValueError("Predictive features contain NaNs for asset rows; check data_pipeline.")

    return df


def main():
    parser = argparse.ArgumentParser( description="Merge feature and label Parquets into model_matrix.parquet")
    parser.add_argument( "--features", type=Path, required=True, help="Input features Parquet (from data_pipeline)")
    parser.add_argument( "--labels", type=Path, required=True, help="Input labels Parquet (from target_generation)")
    parser.add_argument( "--output", type=Path, required=True, help="Path to write merged model matrix Parquet")
    args = parser.parse_args()

    # Merge and validate
    df = merge_and_validate(args.features, args.labels)

    # Persist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, engine="pyarrow", compression="snappy")
    print(
        f"Saved model matrix in: {args.output}\n"
        f"  shape: {df.shape}\n"
        f"  columns: {list(df.columns)}"
    )
    # TODO: begin debug
    print(df.info())
    # end debug


if __name__ == "__main__":
    main()
