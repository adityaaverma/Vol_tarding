"""
CRITICAL ANALYSIS: Entry Price Recording Discrepancy

Potential Issue Identified:
- open_position() uses ADVERSE FILLS (ask for longs) for cash accounting
- StraddlePosition initialization uses entry_price from row (likely mid price)
- This creates a gap between:
  * What was paid in cash (ask price)
  * What P&L tracking assumes (mid price)

This is NOT double counting, but it IS a reconciliation issue.
"""

# EXAMPLE SCENARIO:
# Call option: bid=4.95, ask=5.05, mid=5.00
# Put option:  bid=4.95, ask=5.05, mid=5.00

# Entry Day (Long Straddle):
print("=" * 70)
print("ENTRY DAY ANALYSIS")
print("=" * 70)

entry_mid = 5.00 + 5.00  # = 10.00 (what position.entry_price is set to)
entry_ask = 5.05 + 5.05  # = 10.10 (what cash pays for)

print(f"\n1. Position Object (uses MID):")
print(f"   position.entry_price = {entry_mid} (mid prices)")
print(f"   position.current_price = {entry_mid} (initialized to entry_price)")

print(f"\n2. Cash Account (uses ASK):")
print(f"   premium_flow = -(+1) * {entry_ask} * 100 * 1 = -${entry_ask * 100:.0f}")
print(f"   entry_cost = $65")
print(f"   cash -= ${entry_ask * 100 + 65:.0f}")

print(f"\n3. GAP CREATED:")
print(f"   Position assumes entry at: ${entry_mid}")
print(f"   Cash actually paid at: ${entry_ask}")
print(f"   Discrepancy per contract: ${(entry_ask - entry_mid) * 100:.2f}")

# Mark to Market Day
print("\n" + "=" * 70)
print("MARK TO MARKET DAY")
print("=" * 70)

current_mid = 5.10 + 5.10  # = 10.20 (position marks to this)

print(f"\n1. Position marks to MID:")
print(f"   new_price = {current_mid}")
print(f"   total_pnl_change = (+1) * ({current_mid} - {entry_mid}) * 100 * 1 = ${(current_mid - entry_mid) * 100:.2f}")
print(f"   unrealized_pnl = ${(current_mid - entry_mid) * 100:.2f}")

print(f"\n2. Actual cash position unchanged:")
print(f"   No cash flow yet, position still held")
print(f"   Actual value at mid: ${current_mid * 100:.0f}")
print(f"   Entry cash paid: ${entry_ask * 100:.0f}")
print(f"   Actual profit if sold at mid: ${(current_mid - entry_ask) * 100:.2f}")

print(f"\n3. DISCREPANCY:")
print(f"   Position P&L shows: ${(current_mid - entry_mid) * 100:.2f}")
print(f"   Actual P&L (if liquidated at mid): ${(current_mid - entry_ask) * 100:.2f}")
print(f"   Difference: ${(entry_ask - entry_mid) * 100:.2f}")

# Exit Day
print("\n" + "=" * 70)
print("EXIT DAY ANALYSIS")
print("=" * 70)

exit_bid = 5.15 + 5.15  # = 10.30 (what we receive for selling)

print(f"\n1. Position Object:")
print(f"   current_price (from last mark) = {current_mid}")
print(f"   unrealized_pnl = ${(current_mid - entry_mid) * 100:.2f}")

print(f"\n2. Cash Flow from Close:")
print(f"   close_flow = (+1) * {exit_bid} * 100 * 1 = +${exit_bid * 100:.0f}")
print(f"   close_cost = $65")
print(f"   cash += ${exit_bid * 100 - 65:.0f}")

print(f"\n3. Portfolio P&L Update:")
print(f"   _option_pnl += unrealized_pnl = +${(current_mid - entry_mid) * 100:.2f}")
print(f"   (based on mid prices, not actual cash prices)")

print(f"\n4. Final Accounting:")
print(f"   Total cash flow = -${entry_ask * 100 + 65:.0f} + ${exit_bid * 100 - 65:.0f}")
print(f"                   = ${(exit_bid - entry_ask) * 100 - 130:.2f}")
print(f"   Portfolio reports option_pnl = ${(current_mid - entry_mid) * 100:.2f}")
print(f"   MISMATCH = ${(exit_bid - entry_ask - (current_mid - entry_mid)) * 100:.2f}")

print("\n" + "=" * 70)
print("WHAT THIS MEANS")
print("=" * 70)

print("""
✓ NAV ACCOUNTING IS CORRECT:
  NAV = cash (correctly reflects actual prices paid/received)
  
✓ NO DOUBLE COUNTING:
  Each P&L component is only captured once
  
⚠️  METRIC DISCREPANCY:
  cumulative_option_pnl uses mid prices (position tracking)
  Actual cash P&L = entry_ask → exit_bid
  
  The gap = (entry_ask - entry_mid) + (exit_bid - mid_at_exit)
  This is essentially the bid-ask spread + slippage
  
HOW IT WORKS OUT:
1. open_position() updates cash with ACTUAL ask prices paid
2. StraddlePosition marks to MID prices for P&L tracking
3. close_position() updates cash with ACTUAL bid prices received
4. close_position() captures position.unrealized_pnl (mid-based) as _option_pnl

RESULT:
- NAV = cash (correct, actual entry/exit prices)
- _option_pnl = mid-based P&L (may differ from actual by bid-ask spread)

CONSEQUENCE:
- Performance metrics (based on NAV) are CORRECT
- The "cumulative_option_pnl" metric is BIASED UP/DOWN by the spread
- But this is NOT double counting, just a metric discrepancy

RECOMMENDATION:
To get TRUE option P&L (actual vs mid-based):
  true_option_pnl = exit_cash_proceeds - entry_cash_paid - costs
This would require tracking entry/exit prices separately from position.entry_price
""")
