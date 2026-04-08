"""
vol/metrics.py

Optimized, vectorized volatility metrics for large datasets.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------- low-level helpers ----------

def _as_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr


def safe_log_returns(prices) -> np.ndarray:
    """
    Compute log returns with safe handling for zeros / missing values.
    Returns an array of length N where the first value is NaN.
    """
    px = _as_float_array(prices)
    ret = np.full(px.shape[0], np.nan, dtype=float)

    valid = np.isfinite(px) & (px > 0)
    if valid.sum() < 2:
        return ret

    log_px = np.full_like(px, np.nan, dtype=float)
    log_px[valid] = np.log(px[valid])

    # diff on log series, preserving NaN gaps
    prev = np.roll(log_px, 1)
    ret[1:] = log_px[1:] - prev[1:]
    return ret


def _rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x, copy=False)
    return s.rolling(window=window, min_periods=window).mean().to_numpy(dtype=float)


def _rolling_nanstd(x: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    s = pd.Series(x, copy=False)
    return s.rolling(window=window, min_periods=window).std(ddof=ddof).to_numpy(dtype=float)


def _rolling_nanmedian(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x, copy=False)
    return s.rolling(window=window, min_periods=window).median().to_numpy(dtype=float)


def _rolling_nanquantile(x: np.ndarray, window: int, q: float) -> np.ndarray:
    s = pd.Series(x, copy=False)
    return s.rolling(window=window, min_periods=window).quantile(q).to_numpy(dtype=float)


# ---------- core volatility estimators ----------

def realized_vol(prices, window: int = 20, annualization: int = 252) -> np.ndarray:
    """
    Rolling historical realized vol based on close-to-close log returns.
    Annualized by sqrt(annualization).
    """
    px = _as_float_array(prices)
    r = safe_log_returns(px)
    s = pd.Series(r, copy=False)
    rv = s.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(annualization)
    return rv.to_numpy(dtype=float)


def forward_realized_vol(prices, window: int = 20, annualization: int = 252) -> np.ndarray:
    """
    Forward-looking realized vol label:
    vol over the NEXT 'window' returns, aligned to today's row.
    This is for research/backtest labels only.
    """
    px = _as_float_array(prices)
    r = safe_log_returns(px)

    # Reverse-time rolling std on future returns.
    rev = r[::-1]
    fwd = pd.Series(rev, copy=False).rolling(window=window, min_periods=window).std(ddof=0)
    fwd = fwd.to_numpy(dtype=float)[::-1] * np.sqrt(annualization)

    # First and last window-1 rows should naturally be NaN after alignment
    return fwd


def parkinson_vol(high, low, window: int = 20, annualization: int = 252) -> np.ndarray:
    """
    Parkinson volatility estimator using high-low range.
    Inputs can be arrays or pandas Series.
    """
    h = _as_float_array(high)
    l = _as_float_array(low)
    out = np.full(h.shape[0], np.nan, dtype=float)

    valid = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0) & (h >= l)
    if valid.sum() < window:
        return out

    # Daily Parkinson variance
    daily_var = np.full(h.shape[0], np.nan, dtype=float)
    daily_var[valid] = (np.log(h[valid] / l[valid]) ** 2) / (4.0 * np.log(2.0))

    rv = pd.Series(daily_var, copy=False).rolling(window=window, min_periods=window).mean()
    out = np.sqrt(rv.to_numpy(dtype=float) * annualization)
    return out


def garman_klass_vol(open_, high, low, close, window: int = 20, annualization: int = 252) -> np.ndarray:
    """
    Garman-Klass volatility estimator.
    Requires OHLC arrays.
    """
    o = _as_float_array(open_)
    h = _as_float_array(high)
    l = _as_float_array(low)
    c = _as_float_array(close)
    out = np.full(o.shape[0], np.nan, dtype=float)

    valid = (
        np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
        & (o > 0) & (h > 0) & (l > 0) & (c > 0)
        & (h >= l)
    )
    if valid.sum() < window:
        return out

    log_hl = np.log(h[valid] / l[valid])
    log_co = np.log(c[valid] / o[valid])
    daily_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)

    full_var = np.full(o.shape[0], np.nan, dtype=float)
    full_var[np.where(valid)[0]] = daily_var

    rv = pd.Series(full_var, copy=False).rolling(window=window, min_periods=window).mean()
    out = np.sqrt(rv.to_numpy(dtype=float) * annualization)
    return out


# ---------- rolling feature transforms ----------

def rolling_zscore(x, window: int = 20, ddof: int = 0) -> np.ndarray:
    """
    Rolling z-score using past window only.
    If std == 0, returns NaN at that point.
    """
    arr = _as_float_array(x)
    s = pd.Series(arr, copy=False)
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=ddof)

    z = (s - mean) / std.replace(0.0, np.nan)
    return z.to_numpy(dtype=float)


def ewma_zscore(x, span: int = 20, min_periods: Optional[int] = None) -> np.ndarray:
    """
    Exponentially weighted z-score.
    Helpful when you want more recent observations to dominate.
    """
    arr = _as_float_array(x)
    s = pd.Series(arr, copy=False)
    if min_periods is None:
        min_periods = span

    mean = s.ewm(span=span, adjust=False, min_periods=min_periods).mean()
    var = s.ewm(span=span, adjust=False, min_periods=min_periods).var(bias=False)
    z = (s - mean) / np.sqrt(var.replace(0.0, np.nan))
    return z.to_numpy(dtype=float)


def rolling_percentile(x, window: int = 20) -> np.ndarray:
    """
    Rolling percentile rank of the current value within the trailing window.
    Returns values in [0, 1].
    """
    arr = _as_float_array(x)
    out = np.full(arr.shape[0], np.nan, dtype=float)

    s = pd.Series(arr, copy=False)
    for i in range(window - 1, len(arr)):
        w = s.iloc[i - window + 1 : i + 1].to_numpy(dtype=float)
        cur = w[-1]
        valid = np.isfinite(w)
        if valid.sum() < window or not np.isfinite(cur):
            continue
        out[i] = np.mean(w[valid] <= cur)
    return out


def rolling_minmax_scale(x, window: int = 20) -> np.ndarray:
    """
    Scales current value to [0, 1] over the trailing window.
    """
    arr = _as_float_array(x)
    s = pd.Series(arr, copy=False)
    lo = s.rolling(window=window, min_periods=window).min()
    hi = s.rolling(window=window, min_periods=window).max()
    scaled = (s - lo) / (hi - lo).replace(0.0, np.nan)
    return scaled.to_numpy(dtype=float)


def rolling_mad_zscore(x, window: int = 20) -> np.ndarray:
    """
    Robust z-score using median and MAD.
    More stable than standard z-score in fat-tailed vol data.
    """
    arr = _as_float_array(x)
    s = pd.Series(arr, copy=False)
    med = s.rolling(window=window, min_periods=window).median()
    mad = s.rolling(window=window, min_periods=window).apply(
        lambda a: np.median(np.abs(a - np.median(a))), raw=True
    )
    scale = 1.4826 * mad.replace(0.0, np.nan)
    z = (s - med) / scale
    return z.to_numpy(dtype=float)


# ---------- spreads / diagnostics ----------

def iv_rv_spread(iv, rv) -> np.ndarray:
    """
    Simple IV - RV spread.
    """
    iv = _as_float_array(iv)
    rv = _as_float_array(rv)
    return iv - rv


def spread_zscore(iv, rv, window: int = 20) -> np.ndarray:
    """
    Rolling z-score of the IV-RV spread.
    """
    return rolling_zscore(iv_rv_spread(iv, rv), window=window)


def vol_of_vol(series, window: int = 20, annualization: int = 252) -> np.ndarray:
    """
    Rolling std of a volatility series.
    Useful for regime detection.
    """
    x = _as_float_array(series)
    return pd.Series(x, copy=False).rolling(window=window, min_periods=window).std(ddof=0).to_numpy(dtype=float)


def slope_feature(fast, slow) -> np.ndarray:
    """
    Difference between a fast and slow series.
    """
    f = _as_float_array(fast)
    s = _as_float_array(slow)
    return f - s


def clip_winsor(x, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """
    Winsorize an array using empirical quantiles.
    """
    arr = _as_float_array(x)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return arr
    lo = np.nanquantile(arr[valid], lower_q)
    hi = np.nanquantile(arr[valid], upper_q)
    return np.clip(arr, lo, hi)


def robust_scale(x, window: int = 20) -> np.ndarray:
    """
    Convenience alias for robust z-score.
    """
    return rolling_mad_zscore(x, window=window)
