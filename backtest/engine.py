import pandas as pd
import numpy as np
from strategy.sizing import VolSizer
from execution.delta_hedge import DeltaHedgeEngine, HedgeAction
from strategy.position import PositionManager
from execution.transactions_costs import CostConfig, TransactionalCostModel
from typing import Optional
from dataclasses import dataclass, field
from backtest.portfolio import Portfolio
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital:    float          = 1_000_000.0
    costConfig:         CostConfig     = field(default_factory=CostConfig)
    multiplier:         float          = 100
    ticker:             str            = "SPY"

    # Risk management — fractions of entry value (0.40 = 40%)
    stop_loss_pct:      Optional[float] = 0.40
    profit_take_pct:    Optional[float] = 0.50

    # Preferred DTE window for contract selection at entry
    entry_dte_min:      int            = 25
    entry_dte_max:      int            = 55


class VolBacktest:

    def __init__(
        self,
        signals_df: pd.DataFrame,
        options_df: pd.DataFrame,
        sizer:      VolSizer,
        hedger:     DeltaHedgeEngine,
        config:     BacktestConfig,
    ) -> None:

        self.signals_df = signals_df.sort_values("quote_date").reset_index(drop=True)

        # ── Build the options index ───────────────────────────────────────────
        self.options_df = options_df.copy()
        self.options_df["quote_date"]  = pd.to_datetime(self.options_df["quote_date"])
        self.options_df["expire_date"] = pd.to_datetime(self.options_df["expire_date"])
        self.options_df["strike"]      = self.options_df["strike"].astype(np.float64)
        self.options_df = (
            self.options_df
            .set_index(["quote_date", "strike", "expire_date"])
            .sort_index()
        )

        self.sizer   = sizer
        self.hedger  = hedger

        # FIX 1: never silently fall back to defaults when a real config is passed
        self.config = config if config is not None else BacktestConfig()

        self.cost_model       = TransactionalCostModel(self.config.costConfig)
        self.position_manager = PositionManager(ticker=self.config.ticker)
        self._portfolio       = Portfolio(
            initial_capital = self.config.initial_capital,
            cost_config     = self.config.costConfig,
            multiplier      = self.config.multiplier,
        )
        # FIX 5: last known c_iv/p_iv for the current position, carried forward
        # on NaN rows.  full_chain keeps contracts with NaN IV (the IV solver
        # fails for deep ITM/OTM rows) so MTM rows frequently have NaN IV.
        # Without carry-forward, vega PnL collapses to ~$0 → large Greek residual.
        # Reset to None whenever a new position is opened.
        self._last_valid_iv: Optional[dict] = None
        self.last_valid_contract:Optional[pd.Series]=None

    # ── Type-safe index lookup ─────────────────────────────────────────────────

    def _lookup(
        self,
        date:   pd.Timestamp,
        strike: float,
        expiry,
    ) -> Optional[pd.Series]:
        """
        FIX 2: The root cause of all data-gap warnings.

        pandas MultiIndex .loc[] requires an exact type match on every level.
        pos.expiry from position.py can be datetime.date, np.datetime64, a
        string, or pd.Timestamp depending on how the Position dataclass stores
        it.  Any mismatch silently raises KeyError even when the row exists.

        This wrapper coerces all three key components to the exact types used
        when the index was built (pd.Timestamp / float64), then does the lookup.
        Returns None on a genuine miss so callers can handle it cleanly.
        """
        key = (
            pd.Timestamp(date),      # normalize: datetime.date → Timestamp
            float(strike),           # normalize: float32 / int → float64
            pd.Timestamp(expiry),    # normalize: datetime.date / np.datetime64 → Timestamp
        )
        try:
            row = self.options_df.loc[key]
            # If multiple rows exist for the key (shouldn't happen, but guard it)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.copy()
        except KeyError:
            return None

    def _lookup_daily_chain(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Return all contracts for a given quote_date as a flat DataFrame.
        Coerces the date to Timestamp so xs() always hits the index.
        """
        try:
            chain = (
                self.options_df
                .xs(pd.Timestamp(date), level="quote_date")
                .reset_index()
            )
            return chain
        except KeyError:
            return pd.DataFrame()

    # ── Entry helpers ──────────────────────────────────────────────────────────

    def _select_entry_row(self, daily_chain: pd.DataFrame, spot: float) -> Optional[pd.Series]:
        """
        FIX 3: Original code used .iloc[0] which always picked the shortest
        available expiry — sometimes only 8 DTE — producing very short-lived
        straddles.

        Priority order:
          1. Contracts whose DTE is inside [entry_dte_min, entry_dte_max].
          2. If none qualify, fall back to the nearest-DTE contract.
        """
        if daily_chain.empty:
            return None

        best_strike = self.position_manager.select_strike(daily_chain, spot, method="delta")
        candidates  = daily_chain[daily_chain["strike"] == best_strike].copy()

        if candidates.empty:
            return None

        # Prefer contracts in the target DTE window
        if "dte" in candidates.columns:
            in_window = candidates[
                (candidates["dte"] >= self.config.entry_dte_min) &
                (candidates["dte"] <= self.config.entry_dte_max)
            ]
            if not in_window.empty:
                # Among qualifying contracts, pick the one with DTE closest to
                # the midpoint of the window for maximum theta/vega balance
                mid_dte = (self.config.entry_dte_min + self.config.entry_dte_max) / 2
                idx = (in_window["dte"] - mid_dte).abs().idxmin()
                return in_window.loc[idx].copy()

        # Fallback: shortest available DTE (original behaviour)
        return candidates.iloc[0].copy()

    def _fill_iv(self, contract_data: pd.Series) -> pd.Series:
        """
        FIX 5 (continued): Carry the last known c_iv / p_iv forward when the
        current MTM row has NaN IV.

        Why rows have NaN IV:
          • full_chain keeps every tradeable contract (bid/ask > 0).
          • The IV solver fails for deep-ITM contracts, very wide spreads, or
            contracts close to expiry — those rows have NaN c_iv / p_iv.
          • As a held position ages and drifts away from ATM, more of its
            daily rows will have NaN IV.

        Carry-forward is the simplest defensible approximation: last known IV
        is more informative than zero for Greek attribution.
        """
        c_iv = contract_data.get("c_iv")
        p_iv = contract_data.get("p_iv")

        iv_valid = (
            c_iv is not None and not (isinstance(c_iv, float) and np.isnan(c_iv)) and
            p_iv is not None and not (isinstance(p_iv, float) and np.isnan(p_iv))
        )

        if iv_valid:
            # Update the cache with fresh values
            self._last_valid_iv = {"c_iv": float(c_iv), "p_iv": float(p_iv)}
            return contract_data

        # IV is NaN — patch with last known values if available
        if self._last_valid_iv is not None:
            contract_data = contract_data.copy()
            if pd.isna(contract_data.get("c_iv", np.nan)):
                contract_data["c_iv"] = self._last_valid_iv["c_iv"]
            if pd.isna(contract_data.get("p_iv", np.nan)):
                contract_data["p_iv"] = self._last_valid_iv["p_iv"]

        return contract_data

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        results = []

        for _, signal_row in self.signals_df.iterrows():
            date      = pd.Timestamp(signal_row["quote_date"])
            spot      = float(signal_row.get("underlying_last", 0.0))
            just_exited = False

            # ── 1. Exit Logic ─────────────────────────────────────────────────
            if self._portfolio.position is not None:
                should_exit = (
                    signal_row.get("exit_flag", 0) == 1
                    or self._check_stop()
                )

                if should_exit:
                    pos = self._portfolio.position
                    contract_data = self._lookup(date, pos.strike, pos.expiry)

                    if contract_data is not None:
                        contract_data["underlying_last"] = spot
                    else:
                        # FIX 4: richer fallback — include both legs so
                        # close_position can compute Greek attribution
                        logger.warning(
                            f"Data gap on exit: Strike {pos.strike} / "
                            f"Expiry {pos.expiry} not found on {date.date()}. "
                            f"Closing at entry price."
                        )
                        half = pos.current_price / 2
                        contract_data = pd.Series({
                            "c_bid": half,  "c_ask": half,
                            "p_bid": half,  "p_ask": half,
                            "c_iv":  np.nan, "p_iv": np.nan,
                            "c_delta": 0.0,  "p_delta": 0.0,
                            "c_gamma": 0.0,  "p_gamma": 0.0,
                            "c_vega":  0.0,  "p_vega":  0.0,
                            "c_theta": 0.0,  "p_theta": 0.0,
                            "underlying_last": spot,
                        })
                    if contract_data is not None:
                        contract_data['UNDERLYING_LAST'] = spot
                        contract_data=self._fill_iv(contract_data)
                        self._portfolio.position.mark_to_market(contract_data)
                    self._portfolio.close_position(contract_data)

                    if self._portfolio.shares != 0:
                        self._portfolio.apply_hedge(-self._portfolio.shares, spot)

                    just_exited = True

            # ── 2. Entry Logic ────────────────────────────────────────────────
            if self._portfolio.position is None and signal_row.get("entry_flag", 0) == 1:
                raw_side = int(signal_row.get("position", 0))

                if raw_side not in [1, -1]:
                    logger.warning(f"Skipping entry on {date.date()}: invalid side {raw_side}.")
                else:
                    daily_chain = self._lookup_daily_chain(date)

                    if not daily_chain.empty:
                        entry_row = self._select_entry_row(daily_chain, spot)

                        if entry_row is not None:
                            entry_row = entry_row.copy()
                            entry_row["quote_date"]      = date
                            entry_row["underlying_last"] = spot
                            entry_row["position"]        = raw_side

                            qty = self.sizer.calculate_quantity(
                                entry_row,
                                abs(float(signal_row.get("out", 1.0))),
                            )
                            new_pos = self.position_manager.create_straddle(entry_row, qty)
                            self._last_valid_iv = None
                            self._last_valid_contract = None          # reset IV cache for new position
                            self._fill_iv(entry_row)            # seed cache from entry row if IV present
                            self._portfolio.open_position(new_pos, entry_row)
                        else:
                            logger.warning(f"No valid entry row on {date.date()}.")
                    else:
                        logger.warning(f"No options chain data for entry on {date.date()}.")

            # ── 3. Mark to Market ─────────────────────────────────────────────
            if self._portfolio.position is not None:
                pos = self._portfolio.position
                contract_data = self._lookup(date, pos.strike, pos.expiry)

                if contract_data is not None:
                    contract_data["underlying_last"] = spot
                    contract_data = self._fill_iv(contract_data)   # carry IV forward if NaN
                    self.last_valid_contract=contract_data.copy()
                    snap = self._portfolio.mark_to_market(contract_data)
                elif self.last_valid_contract is not None:
                    # MTM gap: carry last known value, log once per day
                    logger.warning(
                    f"MTM gap on {date.date()} for strike {pos.strike}/{pos.expiry}: "
                    f"carrying forward last known contract data.")
                    carried=self.last_valid_contract.copy()
                    carried["underlying_last"] = spot
                    snap = self._portfolio.mark_to_market(carried)
                else:
                    logger.warning(f"MTM gap on {date.date()}: no prior data to carry forward.")
                    snap = self._portfolio.mark_to_market(pd.Series({"underlying_last": spot}))
            else:
                snap = self._portfolio.mark_to_market(
                    pd.Series({"underlying_last": spot})
                )

            # ── 4. Delta Hedge ────────────────────────────────────────────────
            if self._portfolio.position is not None and not just_exited:
                hedge_action: HedgeAction = self.hedger.calculate_hedge_action(
                    current_shares    = self._portfolio.shares,
                    portfolio_delta   = self._portfolio.position.portfolio_delta,
                    spot              = spot,
                    is_closing        = False,
                )
                self._portfolio.apply_hedge(hedge_action.shares_to_trade, spot)

            snap["date"]   = date
            snap["spot"]   = spot
            snap["signal"] = int(signal_row.get("signal", 0))
            snap["out"]    = float(signal_row.get("out", np.nan))
            results.append(snap)

        results_df = pd.DataFrame(results)
        return self._add_drawdown(results_df)

    # ── Risk guards ────────────────────────────────────────────────────────────

    def _check_stop(self) -> bool:
        if self._portfolio.position is None:
            return False

        pos = self._portfolio.position
        entry_value = self.config.multiplier * pos.entry_price * pos.quantity

        if entry_value == 0:
            return False

        pnl_pct = pos.unrealized_pnl / entry_value

        if self.config.stop_loss_pct is not None and pnl_pct <= -self.config.stop_loss_pct:
            logger.info(
                f"Stop loss triggered at {pnl_pct * 100:.1f}% loss "
                f"(threshold: -{self.config.stop_loss_pct * 100:.0f}%)."
            )
            return True

        if self.config.profit_take_pct is not None and pnl_pct >= self.config.profit_take_pct:
            logger.info(
                f"Profit take triggered at {pnl_pct * 100:.1f}% gain "
                f"(threshold: +{self.config.profit_take_pct * 100:.0f}%)."
            )
            return True

        return False

    @staticmethod
    def _add_drawdown(df: pd.DataFrame) -> pd.DataFrame:
        """Compute peak-to-trough drawdown on NAV."""
        nav  = df["nav"].to_numpy()
        peak = np.maximum.accumulate(nav)
        df["drawdown"] = np.where(peak > 0, (nav - peak) / peak, 0.0)
        return df