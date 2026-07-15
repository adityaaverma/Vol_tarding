import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_pipeline import run_pipeline, load_full_chain, load_signal_data
from strategy.signal import run_signal_pipeline
from strategy.rules import RuleConfig, run_rules
from strategy.sizing import SizerConfig, VolSizer
from execution.delta_hedge import HedgeConfig, DeltaHedgeEngine
from execution.transactions_costs import CostConfig
from backtest.engine import BacktestConfig, VolBacktest
from backtest.performace import compute_performance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("VolRunner")


# ── paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_PATH        = "data/SPY_ALL_YEARS_MASTER.parquet"   # master parquet file
FULL_CHAIN_PATH      = "data/processed/full_chain.parquet"
SIGNAL_DATA_PATH     = "data/processed/signal_data.parquet"


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (or load from cache) the two processed datasets.

    full_chain  — every tradeable contract (bid/ask > 0), DTE 5-120.
                  Used as options_df in the engine so held positions always
                  have a row regardless of how many days you hold them.

    signal_data — ATM-only, DTE 25-55, both IVs present.
                  Used only for signal feature computation.

    On first run this processes the raw CSV and saves parquet files.
    Subsequent runs load the cached parquet in seconds.
    """
    full_chain_exists  = os.path.exists(FULL_CHAIN_PATH)
    signal_data_exists = os.path.exists(SIGNAL_DATA_PATH)

    if full_chain_exists and signal_data_exists:
        logger.info("Cached parquet files found — loading processed data...")
        full_chain  = load_full_chain(FULL_CHAIN_PATH)
        signal_data = load_signal_data(SIGNAL_DATA_PATH)
    else:
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(
                f"Raw data not found at '{RAW_DATA_PATH}'.\n"
                "Ensure SPY_ALL_YEARS_MASTER.parquet is placed in the data/ directory."
            )
        logger.info("No cached files found — running full data pipeline...")
        full_chain, signal_data = run_pipeline(
            raw_path   = RAW_DATA_PATH,
            output_dir = "data/processed",
        )

    logger.info(f"Full chain  : {len(full_chain):,} rows | {full_chain['quote_date'].nunique()} dates")
    logger.info(f"Signal data : {len(signal_data):,} rows | {signal_data['quote_date'].nunique()} dates")
    return full_chain, signal_data


def plot_results(results_df: pd.DataFrame) -> None:
    """
    3-panel performance dashboard:
      1. NAV curve
      2. Option PnL vs Hedge PnL vs Costs (cumulative)
      3. Greek PnL attribution (delta / gamma / vega / theta)
    """
    if results_df.empty:
        logger.warning("No results to plot.")
        return

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2, 1.2, 1.2], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    dates = results_df["date"]

    # ── Panel 1: NAV ──────────────────────────────────────────────────────────
    ax1.plot(dates, results_df["nav"], color="#00FFCC", linewidth=2, label="Total NAV")
    ax1.axhline(results_df["nav"].iloc[0], color="white", linestyle="--", alpha=0.4)
    ax1.fill_between(
        dates,
        results_df["nav"],
        results_df["nav"].iloc[0],
        where=results_df["nav"] >= results_df["nav"].iloc[0],
        alpha=0.08, color="#00FFCC",
    )
    ax1.fill_between(
        dates,
        results_df["nav"],
        results_df["nav"].iloc[0],
        where=results_df["nav"] < results_df["nav"].iloc[0],
        alpha=0.08, color="#FF3366",
    )
    ax1.set_title("Volatility Strategy — Full Performance Dashboard",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Portfolio NAV ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.15)

    # ── Panel 2: PnL bucket attribution ──────────────────────────────────────
    ax2.plot(dates, results_df["cumulative_option_pnl"],
             color="#FF3366", linewidth=1.5, label="Option PnL (Theta/Vega)")
    ax2.plot(dates, results_df["cumulative_hedge_pnl"],
             color="#33CCFF", linewidth=1.5, label="Hedge PnL (Gamma Scalp)")
    ax2.plot(dates, -results_df["cumulative_costs"],
             color="#FFCC00", linewidth=1.5, label="Frictions (–Costs)")
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.4)
    ax2.set_ylabel("Cumulative PnL ($)")
    ax2.set_title("PnL Bucket Attribution", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.15)

    # ── Panel 3: Greek PnL attribution ────────────────────────────────────────
    greek_cols = {
        "cumulative_delta_pnl": ("#FF8C00", "Delta PnL"),
        "cumulative_gamma_pnl": ("#7FFF00", "Gamma PnL"),
        "cumulative_vega_pnl":  ("#DA70D6", "Vega PnL"),
        "cumulative_theta_pnl": ("#87CEEB", "Theta PnL"),
    }
    for col, (color, label) in greek_cols.items():
        if col in results_df.columns:
            ax3.plot(dates, results_df[col], color=color, linewidth=1.5, label=label)

    ax3.axhline(0, color="white", linewidth=0.5, alpha=0.4)
    ax3.set_ylabel("Cumulative PnL ($)")
    ax3.set_xlabel("Date")
    ax3.set_title("Greek PnL Attribution (Taylor Expansion)", fontsize=11)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.15)

    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/backtest_results.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved performance chart to {out_path}")
    plt.show()


def main() -> None:
    logger.info("Initializing Volatility Backtest")

    # ── 1. Data pipeline ──────────────────────────────────────────────────────
    # full_chain  → options_df  (DTE 5-120, bid/ask > 0, no IV filter)
    # signal_data → signal features only (DTE 25-55, ATM, both IVs present)
    #
    # These MUST be separate datasets.  Using the same DTE-filtered file for
    # both causes data gaps the moment a held position ages past the DTE cutoff.
    full_chain, signal_data = prepare_data()

    logger.info("Running Signal Pipeline (IV vs RV extraction)...")
    signals_df = run_signal_pipeline(signal_data)

    logger.info("Applying Trade Rules (Liquidity, Lags, Cooldowns)...")
    # rule_config = RuleConfig(
    #     execution_lag    = 1,
    #     min_liquidity    = 0.5,
    #     allow_flip       = False,
    #     max_holding_days = 21,   # force exit at 3 weeks to avoid DTE → 0 gaps
    #     cooldown_days    = 2,
    # )
    rule_config = RuleConfig(
    execution_lag    = 1,
    min_liquidity    = 0.0,     # ← temporarily disable gate
    allow_flip       = False,
    max_holding_days = 21,
    cooldown_days    = 2,
    )
    executable_signals = run_rules(df=signals_df, config=rule_config)

    # ── 2. Configure strategy ─────────────────────────────────────────────────
    logger.info("Configuring Risk, Sizing, and Hedging Parameters...")

    sizer_cfg = SizerConfig(mode="vega", target_vega_usd=1000.0, max_notional_pct=0.15)
    hedge_cfg = HedgeConfig(mode="BAND", threshold=5.0, rounding=True)
    cost_cfg  = CostConfig(
        option_commission_per_contract = 0.65,
        equity_commission_per_share    = 0.005,
        sec_fee_rate                   = 8.0 / 1_000_000,
        fallback_spread                = 0.05,
    )
    backtest_cfg = BacktestConfig(
        initial_capital  = 1_000_000.0,
        costConfig       = cost_cfg,
        stop_loss_pct    = 0.40,
        profit_take_pct  = 0.50,   # corrected: take profit at 50%, not 100%
    )

    sizer  = VolSizer(config=sizer_cfg)
    hedger = DeltaHedgeEngine(config=hedge_cfg)

    engine = VolBacktest(
        signals_df = executable_signals,
        options_df = full_chain,      # ← full chain, NOT signal_data
        sizer      = sizer,
        config     = backtest_cfg,
        hedger     = hedger,
    )

    # ── 3. Simulation ─────────────────────────────────────────────────────────
    logger.info("Starting Event-Driven Engine. Simulating tick-by-tick...")
    results_df = engine.run()
    logger.info("Backtest complete.")

    if results_df.empty:
        logger.error("Engine returned empty DataFrame. Check signals and data.")
        return

    # ── 4. Tear sheet ─────────────────────────────────────────────────────────
    metrics = compute_performance(results=results_df)

    print("\n" + "=" * 50)
    print("       VOLATILITY STRATEGY TEAR SHEET")
    print("=" * 50)
    print(f"Total Return:        {metrics.get('total_return_pct', 0):.2f}%")
    print(f"CAGR:                {metrics.get('cagr_pct', 0):.2f}%")
    print(f"Ann. Volatility:     {metrics.get('ann_vol_pct', 0):.2f}%")
    print(f"Sharpe Ratio:        {metrics.get('sharpe', 0):.2f}")
    print(f"Sortino Ratio:       {metrics.get('sortino', 0):.2f}")
    calmar = metrics.get("calmar", np.nan)
    print(f"Calmar Ratio:        {calmar:.2f}" if not np.isnan(calmar) else "Calmar Ratio:        N/A")
    print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
    print("-" * 50)
    print(f"Total Option PnL:   ${metrics.get('total_option_pnl', 0):>12,.0f}  (Theta/Vega)")
    print(f"Total Hedge PnL:    ${metrics.get('total_hedge_pnl', 0):>12,.0f}  (Gamma Scalp)")
    print(f"Total Frictions:    ${metrics.get('total_costs', 0):>12,.0f}  (Costs)")
    eff = metrics.get("hedge_efficiency_x", np.nan)
    print(f"Hedge Efficiency:    {f'{eff:.2f}x' if not np.isnan(eff) else 'N/A'}")
    print("-" * 50)
    print(f"Num Trades:          {metrics.get('num_trades', 0)}")
    print(f"Win Rate:            {metrics.get('win_rate_pct', 0):.1f}%")
    pf = metrics.get("profit_factor", np.nan)
    print(f"Profit Factor:       {f'{pf:.2f}' if not np.isnan(pf) else 'N/A'}")
    print(f"Avg Holding Days:    {metrics.get('avg_holding_days', 0):.1f}")
    print("=" * 50 + "\n")

    # ── 5. Greek attribution summary ──────────────────────────────────────────
    greek_cols = [
        "cumulative_delta_pnl",
        "cumulative_gamma_pnl",
        "cumulative_vega_pnl",
        "cumulative_theta_pnl",
    ]
    if all(c in results_df.columns for c in greek_cols):
        last = results_df.iloc[-1]
        g_delta  = last["cumulative_delta_pnl"]
        g_gamma  = last["cumulative_gamma_pnl"]
        g_vega   = last["cumulative_vega_pnl"]
        g_theta  = last["cumulative_theta_pnl"]
        g_total  = g_delta + g_gamma + g_vega + g_theta
        residual = metrics.get("total_option_pnl", 0) - g_total
        res_pct  = (residual / metrics["total_option_pnl"] * 100
                    if metrics.get("total_option_pnl", 0) != 0 else np.nan)

        print("Greek PnL Attribution (end of backtest):")
        print(f"  Delta PnL : ${g_delta:>12,.0f}")
        print(f"  Gamma PnL : ${g_gamma:>12,.0f}")
        print(f"  Vega  PnL : ${g_vega:>12,.0f}")
        print(f"  Theta PnL : ${g_theta:>12,.0f}")
        print(f"  {'─' * 28}")
        print(f"  Greek Total : ${g_total:>10,.0f}")
        print(f"  Option PnL  : ${metrics.get('total_option_pnl', 0):>10,.0f}")
        print(f"  Residual    : ${residual:>10,.0f}  "
              f"({'N/A' if np.isnan(res_pct) else f'{res_pct:.1f}%'})")
        if not np.isnan(res_pct) and abs(res_pct) > 50:
            print()
            print("  ⚠  Residual > 50% — check that c_iv / p_iv columns")
            print("     are not NaN on mark-to-market rows (see engine_patch.py).")
        print()

    # ── 6. Data gap diagnostic ────────────────────────────────────────────────
    if "has_position" in results_df.columns and "signal" in results_df.columns:
        sig_active = results_df["signal"].abs() > 0
        no_pos     = ~results_df["has_position"].astype(bool)
        missed     = int((sig_active & no_pos).sum())
        total_sig  = int(sig_active.sum())
        if total_sig > 0:
            gap_rate = missed / total_sig * 100
            print(f"Data Gap Diagnostic:")
            print(f"  Signal-active bars   : {total_sig}")
            print(f"  Bars without position: {missed}  ({gap_rate:.1f}%)")
            if missed == 0:
                print("  ✓  No data gaps — full_chain covers all held contracts.")
            else:
                print("  ⚠  Gaps remain — verify full_chain DTE range (5-120) covers hold period.")
            print()

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    plot_results(results_df)


if __name__ == "__main__":
    main()