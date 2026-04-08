"""
strategy/signal.py

Daily volatility signal engine for large option-chain datasets.

Design goals:
- Vectorized pandas/numpy operations
- One row per quote_date for downstream backtesting
- No lookahead in live features
- Separate research labels (fwd_rv, fwd_spread) from live signal inputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from vol.metrics import (
    forward_realized_vol,
    realized_vol,
    rolling_percentile,
    rolling_zscore,
    safe_log_returns,
)


@dataclass(frozen=True)
class SignalConfig:
    window: int = 20
    entry_z: float = 1.5
    exit_z: float = 0.5
    min_history: int = 60

    # Weights for composite scoring
    w_spread_z: float = 0.45
    w_spread_pctile: float = 0.15
    w_skew: float = 0.10
    w_term_slope: float = 0.10
    w_regime: float = 0.10
    w_liquidity: float = 0.10

    # Optional robustness settings
    atm_moneyness_band: float = 0.15
    near_dte_max: int = 30
    far_dte_min: int = 31
    far_dte_max: int = 90

    # Signal output thresholds
    long_score: float = -1.5
    short_score: float = 1.5

    # If a feature is missing, use these defaults
    default_liquidity_score: float = 0.5
    default_regime_score: float = 0.0


class VolSignalEngine:
    """
    Builds a daily signal table from option-chain data.

    Required columns:
        - quote_date
        - moneyness
        - c_iv / p_iv
        - underlying_last

    Optional columns:
        - dte, expiration
        - volume, open_interest, bid_ask_spread, traded_value
        - strike, option_type, iv, delta, gamma, vega
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.cfg = SignalConfig(**cfg)
        self.dailyData: pd.DataFrame = pd.DataFrame()
        self._date_col = "quote_date"

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame, col: str = "quote_date") -> pd.DataFrame:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df = df.copy()
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    @staticmethod
    def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _build_daily_atm_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pick the row closest to ATM for each quote_date.
        This is the main reduction step from the option chain to one daily row.
        """
        df = data.copy()
        df = self._ensure_datetime(df, self._date_col)

        required = {self._date_col, "moneyness", "underlying_last"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df.sort_values([self._date_col, "moneyness"], kind="mergesort")
        df["abs_moneyness"] = df["moneyness"].abs()

        # idxmin is vectorized and avoids per-group Python loops.
        idx = df.groupby(self._date_col, sort=False)["abs_moneyness"].idxmin()
        daily = df.loc[idx].copy()

        # A robust daily IV from c_iv/p_iv if both exist.
        has_c = "c_iv" in daily.columns
        has_p = "p_iv" in daily.columns

        if has_c and has_p:
            c = daily["c_iv"].astype(float)
            p = daily["p_iv"].astype(float)
            daily["iv"] = np.where(c.notna() & p.notna(), 0.5 * (c + p), np.where(c.notna(), c, p))
            daily["skew"] = p - c
            daily["call_put_iv_gap"] = c - p
        elif has_c:
            daily["iv"] = daily["c_iv"].astype(float)
            daily["skew"] = np.nan
            daily["call_put_iv_gap"] = np.nan
        elif has_p:
            daily["iv"] = daily["p_iv"].astype(float)
            daily["skew"] = np.nan
            daily["call_put_iv_gap"] = np.nan
        else:
            raise ValueError("Expected at least one of c_iv or p_iv to be present.")

        # Standardize likely numeric fields if present
        for col in ("volume", "open_interest", "bid_ask_spread", "traded_value", "dte", "delta"):
            if col in daily.columns:
                daily[col] = pd.to_numeric(daily[col], errors="coerce")

        return daily.sort_values(self._date_col).reset_index(drop=True)

    def _compute_liquidity_score(self, daily: pd.DataFrame) -> pd.Series:
        """
        A light-weight liquidity proxy that is stable on large datasets.
        Uses a mix of volume, open interest, and inverse spread when available.
        """
        vol_col = self._first_present(daily, ["volume", "opt_volume", "option_volume"])
        oi_col = self._first_present(daily, ["open_interest", "oi"])
        spread_col = self._first_present(daily, ["bid_ask_spread", "spread_ba", "quoted_spread"])

        parts = []

        if vol_col is not None:
            x = np.log1p(pd.to_numeric(daily[vol_col], errors="coerce").astype(float).clip(lower=0))
            parts.append(rolling_zscore(x.to_numpy(), self.cfg.window))

        if oi_col is not None:
            x = np.log1p(pd.to_numeric(daily[oi_col], errors="coerce").astype(float).clip(lower=0))
            parts.append(rolling_zscore(x.to_numpy(), self.cfg.window))

        if spread_col is not None:
            s = pd.to_numeric(daily[spread_col], errors="coerce").astype(float).to_numpy()
            inv = np.where(np.isfinite(s) & (s > 0), 1.0 / s, np.nan)
            parts.append(rolling_zscore(inv, self.cfg.window))

        if not parts:
            return pd.Series(
                np.full(len(daily), self.cfg.default_liquidity_score, dtype=float),
                index=daily.index,
                name="liquidity_score",
            )

        arr = np.vstack([np.asarray(p, dtype=float) for p in parts])
        score = np.nanmean(arr, axis=0)

        # Soft clamp to keep sizing stable.
        score = np.clip(score, -3.0, 3.0)
        # Map into 0..1-ish range for easier interpretation
        score = 1.0 / (1.0 + np.exp(-score))
        return pd.Series(score, index=daily.index, name="liquidity_score")

    def _compute_term_structure_score(self, daily: pd.DataFrame) -> pd.Series:
        """
        Uses DTE buckets if available. Falls back to NaN if no tenor information exists.
        Positive score => contango-like / richer far IV than near IV.
        """
        if "dte" not in daily.columns:
            return pd.Series(np.nan, index=daily.index, name="term_slope")

        dte = pd.to_numeric(daily["dte"], errors="coerce").astype(float)
        iv = pd.to_numeric(daily["iv"], errors="coerce").astype(float)

        near_mask = dte <= self.cfg.near_dte_max
        far_mask = (dte >= self.cfg.far_dte_min) & (dte <= self.cfg.far_dte_max)

        # Use the daily ATM row as a simple proxy; if your raw chain has multiple tenors
        # per date, a separate surface builder should create near/far tenor summaries.
        # Here we derive a smooth proxy from DTE alone.
        # Negative slope when short-dated IV is comparatively elevated.
        term_proxy = np.where(
            np.isfinite(dte) & np.isfinite(iv),
            np.where(near_mask, -iv, np.where(far_mask, iv, 0.0)),
            np.nan,
        )

        # Rolling average of the proxy gives a stable regime indicator
        return pd.Series(
            rolling_zscore(term_proxy, self.cfg.window),
            index=daily.index,
            name="term_slope",
        )

    def _compute_regime_score(self, daily: pd.DataFrame) -> pd.Series:
        """
        Simple regime score:
        - high RV and widening spread volatility => stress
        - calm periods => more favorable for short vol
        If you already have a separate regime model, feed it in as regime_score.
        """
        if "rv" not in daily.columns:
            return pd.Series(self.cfg.default_regime_score, index=daily.index, name="regime_score")

        rv = pd.to_numeric(daily["rv"], errors="coerce").astype(float).to_numpy()
        rv_z = rolling_zscore(rv, self.cfg.window)

        # Higher RV => more stress. Convert to a centered regime score.
        regime = -np.tanh(np.nan_to_num(rv_z, nan=0.0) / 2.0)
        return pd.Series(regime, index=daily.index, name="regime_score")

    def _build_signal_from_features(self, daily: pd.DataFrame) -> pd.DataFrame:
        out = daily.copy()

        out["returns"] = safe_log_returns(out["underlying_last"].to_numpy(dtype=float))

        # Historical RV for the signal itself; forward RV is only for research labels.
        out["rv"] = realized_vol(out["underlying_last"].to_numpy(dtype=float), self.cfg.window)
        out["fwd_rv"] = forward_realized_vol(out["underlying_last"].to_numpy(dtype=float), self.cfg.window)

        out["spread"] = out["iv"].astype(float) - out["rv"].astype(float)
        out["fwd_spread"] = out["iv"].astype(float) - out["fwd_rv"].astype(float)

        out["spread_z"] = rolling_zscore(out["spread"].to_numpy(dtype=float), self.cfg.window)
        out["spread_pctile"] = rolling_percentile(out["spread"].to_numpy(dtype=float), self.cfg.window)

        out["iv_chg_1d"] = out["iv"].astype(float).diff(1)
        out["iv_chg_5d"] = out["iv"].astype(float).diff(5)
        out["rv_chg_5d"] = out["rv"].astype(float).diff(5)

        out["liquidity_score"] = self._compute_liquidity_score(out)
        out["term_slope"] = self._compute_term_structure_score(out)
        out["regime_score"] = self._compute_regime_score(out)

        # Extra features that are cheap and useful
        out["skew_z"] = rolling_zscore(out["skew"].to_numpy(dtype=float), self.cfg.window)

        # Composite score
        spread_pct = out["spread_pctile"].astype(float).to_numpy()
        spread_z = out["spread_z"].astype(float).to_numpy()
        skew_z = out["skew_z"].astype(float).to_numpy()
        term_slope = out["term_slope"].astype(float).to_numpy()
        regime = out["regime_score"].astype(float).to_numpy()
        liq = out["liquidity_score"].astype(float).to_numpy()

        composite = (
            self.cfg.w_spread_z * np.nan_to_num(spread_z, nan=0.0) +
            self.cfg.w_spread_pctile * np.nan_to_num(spread_pct - 0.5, nan=0.0) +
            self.cfg.w_skew * np.nan_to_num(skew_z, nan=0.0) +
            self.cfg.w_term_slope * np.nan_to_num(term_slope, nan=0.0) +
            self.cfg.w_regime * np.nan_to_num(regime, nan=0.0) +
            self.cfg.w_liquidity * np.nan_to_num(liq - 0.5, nan=0.0)
        )
        out["signal_score"] = composite

        # Trade state
        out["signal"] = np.where(
            composite >= self.cfg.short_score, "SHORT_VOL",
            np.where(composite <= self.cfg.long_score, "LONG_VOL", "NO_TRADE")
        )

        # Strength can be used for sizing
        out["signal_strength"] = np.clip(np.abs(composite) / max(self.cfg.entry_z, 1e-9), 0.0, 3.0)

        # Entry / exit flags based on the composite score
        out["entry_flag"] = (
            (out["signal"] != "NO_TRADE") &
            out["signal_score"].notna() &
            (out["signal_strength"] >= 1.0)
        )

        out["exit_flag"] = (
            out["signal_score"].abs() <= self.cfg.exit_z
        )

        # A simple sizing rule (replace in strategy/sizing.py if you prefer)
        size = np.where(
            out["signal"] == "NO_TRADE", 0.0,
            np.where(out["signal_strength"] < 1.0, 0.0,
                     np.where(out["signal_strength"] < 1.5, 0.25,
                              np.where(out["signal_strength"] < 2.0, 0.5, 1.0)))
        )
        out["position_size"] = size

        # Human-readable reason column
        out["reason"] = np.where(
            out["signal"] == "SHORT_VOL",
            "IV rich vs RV with supportive regime/liquidity",
            np.where(
                out["signal"] == "LONG_VOL",
                "IV cheap vs RV with stress-like regime",
                "No strong edge"
            )
        )

        return out

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point.
        Returns one row per date with live-safe features and research labels.
        """
        daily = self._build_daily_atm_rows(data)

        if len(daily) < self.cfg.min_history:
            raise ValueError(
                f"Not enough history: got {len(daily)} days, need at least {self.cfg.min_history}"
            )

        out = self._build_signal_from_features(daily)
        self.dailyData = out.copy()
        return out

    def build_live_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Same as compute_features, but drops future labels for live use.
        """
        out = self.compute_features(data).copy()
        cols_to_drop = [c for c in ["fwd_rv", "fwd_spread"] if c in out.columns]
        return out.drop(columns=cols_to_drop)

    def latest_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Convenience helper: return the last available signal row.
        """
        out = self.compute_features(data)
        return out.iloc[-1]


# Optional backward-compatible alias
volSignalEngine = VolSignalEngine
