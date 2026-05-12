# Double Counting Analysis: option_pnl vs unrealized_opt_pnl

## Data Flow Trace

### 1. StraddlePosition (strategy/position.py)
- **unrealized_pnl**: Cumulative P&L field in position object
  - Initialized: `0.0` in `__post_init__`
  - Updated: `self.unrealized_pnl += total_pnl_change` in `mark_to_market()`
  - Each mark: `total_pnl_change = side * (new_price - current_price) * qty * LOT_SIZE`

### 2. Portfolio (backtest/portfolio.py)
- **_option_pnl**: Cumulative realized option P&L (portfolio level)
  - Initialized: `0.0`
  - Updated ONLY in: `close_position()` → `self._option_pnl += pos.unrealized_pnl`

### 3. mark_to_market() method (backtest/portfolio.py, lines 129-165)

```
if position exists:
    unrealized_opt_pnl = position.unrealized_pnl  (cumulative unrealized from position)
else:
    unrealized_opt_pnl = 0.0

return {
    "cumulative_option_pnl": self._option_pnl + unrealized_opt_pnl
    ...
}
```

## Lifecycle Trace - Example Trade

### Day 1: Entry
- Position created with entry_price = 100
- mark_to_market() called:
  - position.unrealized_pnl = 0 (first mark with same price)
  - cumulative_option_pnl = 0 + 0 = **0**

### Day 2: Holding (price moves to 105)
- mark_to_market() on position:
  - total_pnl_change = 1 * (105 - 100) * qty * 100 = +500
  - position.unrealized_pnl = 0 + 500 = 500
- Portfolio.mark_to_market():
  - unrealized_opt_pnl = 500
  - cumulative_option_pnl = 0 + 500 = **500** ✓

### Day 3: Holding (price moves to 95)
- mark_to_market() on position:
  - total_pnl_change = 1 * (95 - 105) * qty * 100 = -1000
  - position.unrealized_pnl = 500 + (-1000) = -500
- Portfolio.mark_to_market():
  - unrealized_opt_pnl = -500
  - cumulative_option_pnl = 0 + (-500) = **-500** ✓

### Day 4: Exit
**Call sequence in engine.run():**
1. Exit Logic: `close_position(contract_data)`
2. Entry Logic: (skip if no entry signal)
3. Mark to Market: `mark_to_market(contract_data)` ← CALLED AFTER CLOSE

In `close_position()`:
- Reads: `pos.unrealized_pnl = -500`
- Executes: `self._option_pnl += (-500)`
- Result: `self._option_pnl = 0 + (-500) = -500`
- Sets: `self.position = None`

In `mark_to_market()`:
- Since position is None: `unrealized_opt_pnl = 0.0`
- Result: `cumulative_option_pnl = -500 + 0 = **-500** ✓

### Day 5+: After Exit
- No position: `unrealized_opt_pnl = 0.0`
- `cumulative_option_pnl = -500 + 0 = **-500** (unchanged)` ✓

## Potential Double Counting Scenarios

### Scenario A: UNLIKELY - mark_to_market called twice before close
- Engine structure prevents this - mark_to_market only called once per bar in section 3
- close_position happens in section 1, mark_to_market in section 3

### Scenario B: POSSIBLE - If mark_to_market called BEFORE close on same day
- Would cause position.unrealized_pnl to accumulate delta
- Then close_position() would capture that accumulated value
- **Status**: Not found in engine code, but architectural risk

### Scenario C: CONFIRMED SAFE - position object is discarded after close
- After `close_position()`, `self.position = None`
- Old position object with its unrealized_pnl is garbage collected
- Cannot be re-marked or re-closed

## Findings

### ✓ VERIFIED SAFE:
1. **No double accumulation**: unrealized_pnl is only moved once from position to portfolio during close
2. **Correct state transitions**: 
   - Open position: cumulative = realized (0) + unrealized (current)
   - Closed position: cumulative = realized (includes closed) + unrealized (0)
3. **No loss of data**: The incremental mark_to_market updates on StraddlePosition correctly accumulate to get total P&L

### ⚠️  DESIGN NOTE:
- The variable naming `unrealized_opt_pnl` (singular instance) vs cumulative aggregation could be clearer
- The term "cumulative_option_pnl" is accurate: it's the sum of (realized closed positions + unrealized open position)

## Edge Cases Analysis

### Entry + Exit on Same Day
1. `close_position()` → `_option_pnl += old_position.unrealized_pnl`, `position = None`
2. `open_position()` → creates `new_position`, cash updated
3. `mark_to_market()` → marks `new_position` (first mark captures gap to market)
   - `unrealized_opt_pnl = new_position.unrealized_pnl` (new position's P&L)
   - `cumulative_option_pnl = old_pnl + new_pnl` ✓ **Correct**

### Missing Data Scenario
- If contract data missing: `mark_to_market()` called with minimal Series
- Position's `unrealized_pnl` not updated, stays at last known value ✓ **Safe**

### Greeks Decomposition vs Option P&L
- Greek PnLs (`delta_pnl`, `gamma_pnl`, `vega_pnl`, `theta_pnl`) are **attribution only**
- They should approximate but not exceed total option P&L (due to higher-order Greeks)
- They are NOT added to option_pnl again in NAV calculation ✓ **Not double counted**

### Hedge P&L vs Option P&L
- `hedge_pnl` tracks delta-hedging share position (separate from options)
- `option_pnl` tracks option position only
- These are independent components, correctly added in final metrics ✓ **No overlap**

## Recommendation

✅ **NO DOUBLE COUNTING DETECTED**

The logic is architecturally sound:
- Realized P&L from closed positions → `portfolio._option_pnl` (accumulates once)
- Unrealized P&L from current position → `position.unrealized_pnl` (state, not cumulative)
- Total → `_option_pnl + unrealized_pnl` (correctly sums both components)

**Key Safety Features:**
1. Position object is discarded after close (`self.position = None`)
2. `unrealized_pnl` is never accessed again after `close_position()`
3. Each position's P&L is captured exactly once at close
4. `mark_to_market()` returns a dict snapshot, not a cumulative update

**Possible Code Quality Improvements (not bugs):**
- Rename `unrealized_opt_pnl` → `current_position_unrealized_pnl` for clarity
- Add assertion: `final_nav == initial_capital + option_pnl + hedge_pnl - costs`
- Document that greek attribution is decomposition, not additional P&L
