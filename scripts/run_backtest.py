import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading raw market data from {filepath}...")
    df = pd.read_csv(filepath)
    df["quote_date"]  = pd.to_datetime(df["quote_date"])
    df["expire_date"] = pd.to_datetime(df["expire_date"])
    return df


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

    data_path = r"data\dte90.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}.")
        return

    # ── 1. Data pipeline ──────────────────────────────────────────────────────
    options_df = load_data(data_path)

    logger.info("Running Signal Pipeline (IV vs RV extraction)...")
    signals_df = run_signal_pipeline(options_df)

    logger.info("Applying Trade Rules (Liquidity, Lags, Cooldowns)...")
    rule_config = RuleConfig(execution_lag=1, min_liquidity=0.5, allow_flip=False)
    executable_signals = run_rules(df=signals_df, config=rule_config)

    # ── 2. Configure strategy ─────────────────────────────────────────────────
    logger.info("Configuring Risk, Sizing, and Hedging Parameters...")

    sizer_cfg = SizerConfig(mode="vega", target_vega_usd=1000.0, max_notional_pct=0.15)
    hedge_cfg  = HedgeConfig(mode="BAND", threshold=5.0, rounding=True)
    cost_cfg   = CostConfig(
        option_commission_per_contract=0.65,
        equity_commission_per_share=0.005,
        sec_fee_rate=8.0 / 1_000_000,
        fallback_spread=0.05,
    )
    backtest_cfg = BacktestConfig(
        initial_capital=1_000_000.0,
        costConfig=cost_cfg,
        stop_loss_pct=0.40,
        profit_take_pct=1.00,
    )

    sizer  = VolSizer(config=sizer_cfg)
    hedger = DeltaHedgeEngine(config=hedge_cfg)
    engine = VolBacktest(
        signals_df=executable_signals,
        options_df=options_df,
        sizer=sizer,
        config=backtest_cfg,
        hedger=hedger,
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
    if all(c in results_df.columns for c in [
        "cumulative_delta_pnl", "cumulative_gamma_pnl",
        "cumulative_vega_pnl", "cumulative_theta_pnl",
    ]):
        print("Greek PnL Attribution (end of backtest):")
        print(f"  Delta PnL : ${results_df['cumulative_delta_pnl'].iloc[-1]:>12,.0f}")
        print(f"  Gamma PnL : ${results_df['cumulative_gamma_pnl'].iloc[-1]:>12,.0f}")
        print(f"  Vega  PnL : ${results_df['cumulative_vega_pnl'].iloc[-1]:>12,.0f}")
        print(f"  Theta PnL : ${results_df['cumulative_theta_pnl'].iloc[-1]:>12,.0f}")
        print()

    plot_results(results_df)


if __name__ == "__main__":
    main()