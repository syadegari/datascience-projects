import numpy as np
import pandas as pd


def prob_to_position(p, thresh=0.5):
    return np.clip((p - thresh) / (1 - thresh), 0, 1)


def run_backtest(df: pd.DataFrame, y_prob, lookback: int = 40):
    """
    df       : split model-matrix that still contains ALL tickers
    y_prob   : predicted probabilities for the asset rows *after* warm-up
    lookback : same look-back you used when building the dataset
    """
    # 1. keep asset rows only
    df_asset = df[df["ticker_type"] == "asset"]

    # 2. drop warm-up rows per ticker
    df_asset = (
        df_asset
        .groupby(level="Ticker", group_keys=False)
        .apply(lambda g: g.iloc[lookback-1:])
    )

    # 3. deterministic order: (Ticker, Datetime) then sort
    df_ordered = df_asset.swaplevel().sort_index()

    # 4. sanity-check length
    if len(df_ordered) != len(y_prob):
        raise ValueError(
            f"row mismatch: df has {len(df_ordered)}, "
            f"but y_prob has {len(y_prob)}"
        )

    # 5. P&L vectorisation
    pos = prob_to_position(y_prob)                       # [n_samples]
    pnl_per_row = pos * df_ordered["log_ret"].to_numpy() # [n_samples]

    daily_ret = (
        pd.Series(pnl_per_row, index=df_ordered.index.get_level_values(1))
        .groupby(level=0)
        .mean()
    )
    return daily_ret.cumsum()
