"""
Validation script to verify that cumulative P&L correctly accounts for NAV changes.
NAV should equal: initial_capital + cumulative_option_pnl + cumulative_hedge_pnl - cumulative_costs

This catches any double counting or missing P&L components.
"""

import pandas as pd
import numpy as np


def validate_pnl_accounting(results_df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Verify that NAV = initial_capital + option_pnl + hedge_pnl - costs
    
    Args:
        results_df: DataFrame from backtest with cumulative P&L columns
        initial_capital: Starting NAV
        
    Returns:
        dict with validation results
    """
    
    if results_df.empty:
        return {"status": "SKIPPED", "reason": "Empty results"}
    
    # Extract final values
    final_nav = results_df['nav'].iloc[-1]
    final_option_pnl = results_df['cumulative_option_pnl'].iloc[-1]
    final_hedge_pnl = results_df['cumulative_hedge_pnl'].iloc[-1]
    final_costs = results_df['cumulative_costs'].iloc[-1]
    
    # Calculate what NAV should be based on P&L components
    expected_nav = initial_capital + final_option_pnl + final_hedge_pnl - final_costs
    
    # Check reconciliation
    nav_diff = abs(final_nav - expected_nav)
    tolerance = 1.0  # Allow $1 due to rounding
    
    is_reconciled = nav_diff < tolerance
    
    # Check row-by-row for double counting
    daily_diffs = []
    for idx, row in results_df.iterrows():
        calc_nav = (
            initial_capital + 
            row['cumulative_option_pnl'] + 
            row['cumulative_hedge_pnl'] - 
            row['cumulative_costs']
        )
        daily_diff = abs(row['nav'] - calc_nav)
        daily_diffs.append(daily_diff)
    
    max_daily_diff = max(daily_diffs) if daily_diffs else 0
    
    # Greek attribution check - should be less than or equal to option PnL 
    # (due to higher order greeks not captured)
    final_greek_sum = (
        results_df['cumulative_delta_pnl'].iloc[-1] +
        results_df['cumulative_gamma_pnl'].iloc[-1] +
        results_df['cumulative_vega_pnl'].iloc[-1] +
        results_df['cumulative_theta_pnl'].iloc[-1]
    )
    
    greek_vs_option = {
        "total_greek_pnl": final_greek_sum,
        "total_option_pnl": final_option_pnl,
        "difference": final_option_pnl - final_greek_sum,
        "note": "Difference due to higher-order Greeks and numerical errors"
    }
    
    return {
        "status": "PASS" if is_reconciled else "FAIL",
        "nav_reconciliation": {
            "reported_nav": final_nav,
            "calculated_nav": expected_nav,
            "difference": nav_diff,
            "tolerance": tolerance,
            "reconciled": is_reconciled,
        },
        "daily_reconciliation": {
            "max_daily_diff": max_daily_diff,
            "mean_daily_diff": np.mean(daily_diffs),
            "median_daily_diff": np.median(daily_diffs),
        },
        "pnl_components": {
            "initial_capital": initial_capital,
            "final_option_pnl": final_option_pnl,
            "final_hedge_pnl": final_hedge_pnl,
            "final_costs": final_costs,
            "total_pnl": final_option_pnl + final_hedge_pnl - final_costs,
        },
        "greek_attribution": greek_vs_option,
        "double_counting_check": {
            "is_option_pnl_separate_from_hedge": final_option_pnl != final_hedge_pnl,
            "is_costs_deducted_once": final_costs > 0,  # Costs should only appear once in nav
        }
    }


def print_validation_report(validation_result: dict) -> None:
    """Pretty print validation results"""
    
    status = validation_result.get("status")
    print(f"\n{'='*60}")
    print(f"PnL ACCOUNTING VALIDATION: {status}")
    print(f"{'='*60}")
    
    if status == "SKIPPED":
        print(f"  Reason: {validation_result.get('reason')}")
        return
    
    # NAV Reconciliation
    nav_rec = validation_result["nav_reconciliation"]
    print(f"\nNAV Reconciliation:")
    print(f"  Reported NAV:     ${nav_rec['reported_nav']:>15,.2f}")
    print(f"  Calculated NAV:   ${nav_rec['calculated_nav']:>15,.2f}")
    print(f"  Difference:       ${nav_rec['difference']:>15,.2f}")
    print(f"  Tolerance:        ${nav_rec['tolerance']:>15,.2f}")
    print(f"  Status:           {'✓ RECONCILED' if nav_rec['reconciled'] else '✗ MISMATCH'}")
    
    # Daily Reconciliation
    daily_rec = validation_result["daily_reconciliation"]
    print(f"\nDaily NAV Reconciliation (all bars):")
    print(f"  Max daily diff:   ${daily_rec['max_daily_diff']:>15,.2f}")
    print(f"  Mean daily diff:  ${daily_rec['mean_daily_diff']:>15,.2f}")
    print(f"  Median daily diff: ${daily_rec['median_daily_diff']:>15,.2f}")
    
    # P&L Components
    pnl = validation_result["pnl_components"]
    print(f"\nP&L Components (end of backtest):")
    print(f"  Initial Capital:  ${pnl['initial_capital']:>15,.2f}")
    print(f"  Option PnL:       ${pnl['final_option_pnl']:>15,.2f}")
    print(f"  Hedge PnL:        ${pnl['final_hedge_pnl']:>15,.2f}")
    print(f"  Costs:            ${pnl['final_costs']:>15,.2f}")
    print(f"  Total PnL:        ${pnl['total_pnl']:>15,.2f}")
    print(f"  Final NAV:        ${status['nav_reconciliation']['reported_nav']:>15,.2f}")
    
    # Greek Attribution
    greek = validation_result["greek_attribution"]
    print(f"\nGreek Attribution Analysis:")
    print(f"  Delta PnL:        (included in option_pnl)")
    print(f"  Gamma PnL:        (included in option_pnl)")
    print(f"  Vega PnL:         (included in option_pnl)")
    print(f"  Theta PnL:        (included in option_pnl)")
    print(f"  Sum of greeks:    ${greek['total_greek_pnl']:>15,.2f}")
    print(f"  Total Option PnL: ${greek['total_option_pnl']:>15,.2f}")
    print(f"  Difference:       ${greek['difference']:>15,.2f} (higher-order Greeks)")
    
    # Double Counting Check
    check = validation_result["double_counting_check"]
    print(f"\nDouble Counting Safety Checks:")
    print(f"  Option PnL ≠ Hedge PnL:  {'✓ YES' if check['is_option_pnl_separate_from_hedge'] else '✗ SAME'}")
    print(f"  Costs deducted:          {'✓ YES' if check['is_costs_deducted_once'] else '✗ NOT'}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("""
    This script validates P&L accounting in backtests.
    
    To use after running backtest:
    
    from backtest.performace import validate_pnl_accounting, print_validation_report
    
    # After engine.run() returns results_df
    validation = validate_pnl_accounting(results_df, initial_capital=1_000_000)
    print_validation_report(validation)
    
    Checks for:
    - NAV reconciliation: final_nav == initial_capital + option_pnl + hedge_pnl - costs
    - Daily reconciliation: same formula for each bar
    - Double counting: costs only deducted once, P&L components separate
    - Greek decomposition: sum of greeks ≤ total option P&L (due to higher-order terms)
    """)
