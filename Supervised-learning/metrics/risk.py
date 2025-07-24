"""
ml/metrics/risk.py
==================

Classical portfolio-level risk metrics used in quant back-tests.

Functions
---------
sharpe(daily_ret, risk_free=0.0)
    Annualised Sharpe ratio on daily return series.

max_drawdown(cum_pnl)
    Maximum peak-to-trough drawdown on cumulative P&L.

calmar_ratio(cum_pnl, years=None)
    Annualised return divided by max drawdown.

Sortino, Volatility, and other ratios can be added later.
"""

import numpy as np
import pandas as pd


def _as_series(arr: np.ndarray | pd.Series) -> pd.Series:
    if isinstance(arr, pd.Series):
        return arr
    return pd.Series(arr)


def sharpe(
    daily_ret: np.ndarray | pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Parameters
    ----------
    daily_ret : array-like
        **Daily arithmetic returns** (not cumulative P&L).
    risk_free : float, optional
        Daily risk-free rate; default 0.
    periods_per_year : int, optional
        252 for trading days.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    r = _as_series(daily_ret) - risk_free
    return np.sqrt(periods_per_year) * r.mean() / r.std(ddof=0)


def max_drawdown(cum_pnl: np.ndarray | pd.Series) -> float:
    """
    Parameters
    ----------
    cum_pnl : array-like
        Cumulative P&L series (level, not return).

    Returns
    -------
    float
        Maximum drawdown expressed as **negative** number.
    """
    s = _as_series(cum_pnl)
    peak = s.cummax()
    drawdown = s - peak
    return drawdown.min()


def calmar_ratio(cum_pnl: np.ndarray | pd.Series, years: float | None = None) -> float:
    """
    Annualised Calmar ratio = CAGR / |MaxDD|.

    Parameters
    ----------
    cum_pnl : array-like
        Cumulative P&L series.
    years : float, optional
        Number of years in the series; if None it is inferred
        from index length assuming 252 trading days.

    Returns
    -------
    float
    """
    s = _as_series(cum_pnl)
    total_ret = s.iloc[-1] - s.iloc[0]
    if years is None:
        years = len(s) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1
    mdd = abs(max_drawdown(s))
    return cagr / mdd if mdd != 0 else np.nan
