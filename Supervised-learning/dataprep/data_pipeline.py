import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from logging import Logger

import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ['APD', 'MKTX', 'BRO', 'BLK', 'KHC', 'TDY', 'MDLZ', 'AMP', 'HSY',
       'LYB', 'USB', 'CSX', 'JNJ', 'PGR', 'ADP', 'NCLH', 'CME', 'EW',
       'DPZ', 'BIIB', 'HPE', 'EG', 'LLY', 'CPAY', 'CHD', 'ABBV', 'TXT',
       'NVDA', 'MO', 'IFF', 'APO', 'PEP', 'PAYC', 'MOH', 'CCL', 'WEC',
       'BA', 'GNRC', 'TER', 'WY', 'CCI', 'DIS', 'AXON', 'FICO', 'NSC',
       'WM', 'ITW', 'F', 'SYY', 'XEL', 'CRM', 'NOC', 'FSLR', 'TSLA',
       'ECL', 'NTAP', 'FE', 'ZBRA', 'BALL', 'PODD', 'DTE', 'EXPE', 'NVR',
       'EQR', 'NDAQ', 'GDDY', 'BAC', 'KO', 'WMT', 'GPN', 'LW', 'HST',
       'UDR', 'PEG', 'ALLE', 'INCY', 'SCHW', 'BMY', 'PSX', 'AES', 'RJF',
       'KIM', 'CAG', 'AXP', 'KKR', 'HES', 'MMC', 'AEE', 'FDS', 'FFIV',
       'MHK', 'CTSH', 'CPB', 'GPC', 'BDX', 'EXR', 'WAB', 'INVH', 'MU',
       ]

"""
- Communication Services: XLC
- Consumer Discretionary: XLY
- Consumer Staples: XLP
- Energy: XLE
- Financials: XLF
- Health Care: XLV
- Industrials: XLI
- Information Technology: XLK
- Materials: XLB
- Real Estate: XLRE
- Utilities: XLU
"""
SECTOR_ETFS = {
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU",
}

"""
Market Indicators Tickers:
- Oil:          CL=F     (Crude Oil Futures)
- Gold:         GC=F     (Gold Futures)
- Dollar:       DX=F     (US Dollar Index Futures)
- VIX:          ^VIX     (CBOE Volatility Index)
- 2-Year Yield: ZT=F     (2-Year Treasury Note Futures)
- 10-Year Yield:ZN=F     (10-Year Treasury Note Futures)
- 30-Year Yield:ZB=F     (30-Year Treasury Bond Futures)
"""
MACRO_TICKERS = {
    "CL=F", "GC=F", "DX=F", "^VIX", "ZT=F", "ZN=F", "ZB=F",
}


@dataclass
class Config:
    """Runtime configuration for pipeline execution."""
    forward_fill_missing_asset: bool        # forward fill missing (0 or nan) asset values or remove them entirely from modeling
    read_from_disk: bool                    # read raw values from disk or download them
    assets: list[str]                       # primary tradable equities
    auxiliaries: list[str]                  # supporting tickers (sector ETFs, futures, etc.)
    start: str                              # ISO date or YYYY‑MM‑DD
    end: str                                # ISO date or "today"
    interval: str                           # yfinance interval (e.g. "1d", "30m", "1wk")
    rolling_norm_window: int = 20           # periods for z‑score / vol scaling
    include_sector_etf_volume: bool = True  # toggle for A/B tests
    out_dir: Path = Path("./data")

    def date_range(self) -> tuple[dt.datetime, dt.datetime]:
        def ensure_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
            return timestamp if timestamp.tzinfo else timestamp.tz_localize("UTC")

        start_dt = ensure_utc(pd.Timestamp(self.start))
        end_value = dt.datetime.now(dt.timezone.utc) if self.end == "today" else self.end
        end_dt = ensure_utc(pd.Timestamp(end_value))
        return start_dt, end_dt

    def ticker_types(self) -> dict[str, str]:
        """Return mapping: ticker → {asset|sector_etf|macro}."""
        mapping = {}
        for t in self.assets:
            mapping[t] = "asset"
        for t in self.auxiliaries:
            if t in SECTOR_ETFS:
                mapping[t] = "sector_etf"
            else:
                mapping[t] = "macro"
        return mapping

    def __post_init__(self):
        print("Initialized Config with:")
        for field_name, value in self.__dict__.items():
            if field_name == "assets" or field_name == "auxiliaries":
                print(f"  {field_name}: [omitted, {len(value)} items]")
            else:
                print(f"  {field_name}: {value}")


CONFIG = Config(
    forward_fill_missing_asset=False,
    read_from_disk=True,
    assets=ASSETS,
    auxiliaries=list(SECTOR_ETFS | MACRO_TICKERS),
    start="2018-09-01",
    end="2022-01-01",
    interval="1d",
)


def _download(tickers: Iterable[str], start: str, end: str, interval: str) -> pd.DataFrame:
    return yf.download(
        list(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True)


def set_logger(log_file: str) -> Logger:
    logger = logging.getLogger('missing_logger')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # reset handlers
    handler_console = logging.StreamHandler()
    handler_file = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler_console.setFormatter(formatter)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_console)
    logger.addHandler(handler_file)
    return logger


def handle_missing_asset(df: pd.DataFrame, logger: Logger, forward_fill_asset: bool) -> pd.DataFrame:
    """
    Handle missing or zero values in Close or Volume columns of a yfinance DataFrame with MultiIndex columns.

    Args:
        df (pd.DataFrame): DataFrame with MultiIndex columns (Price, Ticker).
        logger (Logger): Logger for logging missing entries.
        forward_fill_asset (bool): 
            If True, forward-fill missing and zero values.
            If False, drop tickers with any missing or zero values.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    price_fields = ['Close', 'Volume']
    tickers = df.columns.levels[1]
    tickers_to_drop = set()

    for ticker in tickers:
        ticker_had_issues = False
        for price_field in price_fields:
            col = (price_field, ticker)
            if col not in df.columns:
                continue

            series = df[col]
            is_nan = series.isna()
            is_zero = (series == 0)
            missing = is_nan | is_zero

            for date in series[is_nan].index:
                logger.info(f"Missing value (NaN) for {ticker} at {date.date()} in {price_field}")
            for date in series[is_zero].index:
                logger.info(f"Missing value (0) for {ticker} at {date.date()} in {price_field}")

            if missing.any():
                ticker_had_issues = True

        if ticker_had_issues:
            if forward_fill_asset:
                logger.info(f"Action: replacing the missing values with nan for {ticker}\n")
                for price_field in price_fields:
                    col = (price_field, ticker)
                    if col in df.columns:
                        df[col] = df[col].replace(0, np.nan).ffill()
            else:
                logger.info(f"Action: Dropping ticker {ticker} from dataset\n")
                tickers_to_drop.add(ticker)

    if not forward_fill_asset and tickers_to_drop:
        df.drop(columns=tickers_to_drop, level='Ticker', inplace=True)

    return df


def _download_tickers(
    assets: Iterable[str],
    auxiliaries: Iterable[str],
    *,
    start: str,
    end: str,
    interval: str,
    forward_fill_asset: bool,
    read_from_disk: bool
) -> pd.DataFrame: 
    """Fetch *Close* & *Volume* for each ticker and return a tidy
    `(Datetime, Ticker)` indexed frame.

    Notes
    -----
    * yfinance, when **not** using `group_by="ticker"`, returns a MultiIndex
      on *columns* where **level 0 = field (Open/High/…); level 1 = ticker**.
      Stacking `level=1` therefore puts the *ticker* into the row index and
      already yields the desired `(Datetime, Ticker)` order, no need for
      the swap/reorder dance we had earlier.
    * We still call `.sort_index()` to keep the rows chronologically ordered
      within each ticker.
    """
    if read_from_disk:
        raw_asset = pd.read_parquet('../data/raw_asset.parquet', engine="pyarrow")
        raw_aux = pd.read_parquet('../data/raw_aux.parquet', engine="pyarrow")
    else:
        raw_asset = _download(assets, start=start, end=end, interval=interval)[['Volume', 'Close']]
        raw_aux = _download(auxiliaries, start=start, end=end, interval=interval)[['Volume', 'Close']]
        #
        raw_asset.to_parquet('../data/raw_asset.parquet', engine="pyarrow", compression="snappy")
        raw_aux.to_parquet('../data/raw_aux.parquet', engine="pyarrow", compression="snappy")
    
    logger = set_logger('../data/missing_logger')
    raw_asset = handle_missing_asset(raw_asset, logger, forward_fill_asset)

    raw = raw_asset.join(raw_aux, how='left')

    # mixing macro tickers such as oil and gold, which trade on holidays, with other assets,
    # such as stocks and etfs, that don't trade on holidays, will leave nans in places where
    # we have holidays. We prune these out early before flattening and later adding features. 
    # otherwise we'll end up with nans all over the place

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi‑ticker: columns (Field, Ticker) -> stack ticker (level=1)
        raw = raw.stack(level=1, future_stack=True)
    else:
        # Single‑ticker: append the ticker as a new second index level
        raw["Ticker"] = tickers[0]
        raw = raw.set_index("Ticker", append=True)

    raw.index.names = ["Datetime", "Ticker"]
    raw = raw.sort_index()

    # because of above transformation we should have the same number of tickers for each date now
    arr_temp = []
    for date_index in raw.index.levels[0]:
        arr_temp.append(raw.loc[date_index].shape[0])
    assert (np.array(arr_temp) == arr_temp[0]).all(), \
    "All tickers (asset/ETF/macro) should be present in all dates"

    return raw


def _compute_log_returns(close: pd.Series) -> pd.Series:
    ret  = np.log(close).groupby(level=1).diff(1)

    n_inf = np.isinf(ret).sum()
    if n_inf > 0:
        pct = 100 * n_inf / len(ret)
        print(f"[WARN] log_ret: converted {n_inf:,} +-inf ({pct:.5f}%) to NaN (price=0 glitch)")
    ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    return ret

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.groupby(level=1).transform(lambda x: x.rolling(window, min_periods=window//2).mean())
    std  = series.groupby(level=1).transform(lambda x: x.rolling(window, min_periods=window//2).std())

    z = (series - mean) / std

    n_inf = np.isinf(z).sum()
    if n_inf > 0:
        pct = 100 * n_inf / len(z)
        print(f"[WARN] z_log_ret: std=0 produced {n_inf:,} +-inf ({pct:.5f}%) → NaN")
    z.replace([np.inf, -np.inf], np.nan, inplace=True)
    return z


def _scale_volume(volume: pd.Series, window: int) -> pd.Series:
    norm = volume / volume.groupby(level=1).transform(lambda x: x.rolling(window, min_periods=window // 2).mean())
    scaled = np.log(norm)

    n_inf = np.isinf(scaled).sum()

    if n_inf > 0:
        pct = 100 * n_inf / len(scaled)
        print(f"[WARN] _scale_volume: converted {n_inf} +-inf values "
              f"to NaN ({pct:.4f}% of rows) - likely zero volume days."
              "These are going to be masked later. Model will see vol_scaled=0 & mask=1.")

    scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
    return scaled


def lag_features(df: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
    return df.groupby(level=1).shift(lags)


def build_feature_set(cfg: Config) -> pd.DataFrame:
    start_dt, end_dt = cfg.date_range()
    all_tickers = cfg.assets + cfg.auxiliaries

    raw = _download_tickers(
        cfg.assets,
        cfg.auxiliaries,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=cfg.interval,
        forward_fill_assets=cfg.forward_fill_missing_asset,
        read_from_disk=cfg.read_from_disk
    )

    # Null volume for macro or excluded sector ETFs
    types = cfg.ticker_types()
    for tkr, t_type in types.items():
        if t_type == "macro" or (t_type == "sector_etf" and not cfg.include_sector_etf_volume):
            raw.loc[(slice(None), tkr), "Volume"] = np.nan

    # Predictive features (to be lagged)
    pred = pd.DataFrame(index=raw.index)
    pred["log_ret"] = _compute_log_returns(raw["Close"])
    pred["z_log_ret"] = _rolling_zscore(pred["log_ret"], cfg.rolling_norm_window)
    pred["vol_scaled"] = _scale_volume(raw["Volume"], cfg.rolling_norm_window)

    pred = lag_features(pred, lags=1)

    # Combine with raw Close (unlagged) for target generation
    features = pd.concat([pred, raw["Close"]], axis=1)

    features['ticker_type'] = features.index.get_level_values('Ticker').map(types)
    features['ticker_type'] = features['ticker_type'].astype('category')

    features.attrs["ticker_type"] = types

    # Drop rows where predictive features are NaN (first row post‑lag)
    # TODO: Clarify if this interfere with Nan-ing when constructing log_ret
    #       and z_log_ret
    features = features.dropna(subset=["log_ret", "z_log_ret"])

    return features


def save_feature_set(df: pd.DataFrame, cfg: Config) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    fname = cfg.out_dir / f"features_{cfg.interval}_{cfg.start}_{cfg.end}.parquet"
    df.to_parquet(fname, engine="pyarrow", compression="snappy")
    return fname


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run data preparation pipeline")
    ap.add_argument("--interval", default=CONFIG.interval, help="yfinance interval (e.g. 1d, 30m, 1wk)")
    ap.add_argument("--start", default=CONFIG.start, help="ISO start date")
    ap.add_argument("--end", default=CONFIG.end, help="ISO end date or 'today'")
    ap.add_argument("--no_etf_volume", action="store_true", help="Exclude sector‑ETF volume features")
    args = ap.parse_args()

    cfg = CONFIG
    cfg.interval = args.interval
    cfg.start = args.start
    cfg.end = args.end
    cfg.include_sector_etf_volume = not args.no_etf_volume

    feat = build_feature_set(cfg)
    path = save_feature_set(feat, cfg)
    print(f"Saved in: {path} | shape={feat.shape} | ETF vol kept: {cfg.include_sector_etf_volume}")
