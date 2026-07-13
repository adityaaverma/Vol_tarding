# Volatility Options Trading Project Summary (for Claude)

## 1. PROJECT OVERVIEW

### Purpose
This repository appears to be a Python-based volatility/options trading research and backtesting framework focused on SPY options. The main idea is to:
- estimate or ingest implied volatility (IV) and realized volatility (RV)
- generate trading signals based on IV/RV dislocations and related features
- build/execute a backtest around a volatility strategy
- simulate delta hedging and transaction costs
- report performance metrics and plots

This is not a live broker-integrated production system; it is primarily an event-driven research/backtest engine with a strong options-pricing and Greeks focus.

### High-level architecture
The project is organized around a pipeline:
1. Data ingestion and cleaning
2. Signal generation from IV/RV features
3. Rule filtering (liquidity, lag, cooldown, max holding)
4. Position sizing and contract selection
5. Backtest engine with mark-to-market, hedging, and cost modeling
6. Performance reporting and visualization

Major modules:
- Data layer: loading and preprocessing option chain data
- Strategy layer: signal generation, trade rules, sizing, position management
- Execution layer: delta hedging and transaction cost logic
- Backtest layer: portfolio, performance, and event simulation
- Pricing layer: Black-Scholes pricing and implied vol solvers

---

## 2. FOLDER STRUCTURE

```text
Vol_tarding/
├── backtest/
│   ├── engine.py          # main backtest loop and event-driven simulation
│   ├── performance.py     # return, Sharpe, drawdown, trade stats
│   ├── portfolio.py       # cash/position ledger and NAV tracking
│   └── validate_accounting.py
├── bs/
│   ├── greeks.py          # Black-Scholes Greeks
│   ├── implied_vol.py    # IV solver (Newton + bisection)
│   └── pricing.py         # Black-Scholes option pricing
├── data/
│   ├── data_pipeline.py   # main parquet/CSV cleaning and feature prep
│   ├── loaders.py         # Yahoo/CBOE option-chain fetchers
│   ├── processed/         # cached parquet outputs
│   └── SPY_ALL_YEARS_MASTER.parquet
├── execution/
│   ├── delta_hedge.py     # hedge engine
│   └── transactions_costs.py
├── logs/                  # runtime logs / charts
├── plots/                 # visualization artifacts
├── quant/                 # local Python virtualenv
├── scripts/
│   ├── run_backtest.py    # main entry point
│   ├── build_iv_surface.py
│   ├── diagnose.py
│   ├── live_surface.py
│   └── vol_try.ipynb
├── strategy/
│   ├── position.py        # straddle position and strike selection
│   ├── rules.py           # trade/execution rules
│   ├── signal.py          # signal engine (older / research version)
│   └── sizing.py          # position sizing logic
├── visualization/
│   └── plots.py           # plotting utilities
├── vol/
│   ├── interpolation.py   # IV surface interpolation helpers
│   ├── iv_surface.py      # IV solver for chains
│   ├── metrics.py         # volatility metrics
│   ├── realized_vol.py    # RV calculations
│   └── metrcis(gpt).py    # alternate/experimental metrics module
├── requirements.txt
├── singal.py              # legacy / experimental signal module
└── vol_trading_system_flowchart.html
```

---

## 3. KEY FILES & THEIR RESPONSIBILITY

### scripts/run_backtest.py
- Main orchestrator for the backtest.
- Functions/classes: prepare_data(), plot_results(), main().
- It loads processed parquet data, runs signal generation, applies rules/sizing/hedging, executes the backtest, prints a performance tear sheet, and saves a chart.

### data/data_pipeline.py
- Handles ingestion and normalization of raw option-chain data.
- Key helpers: inspect_schema(), load_raw(), _load_parquet_batched(), _resolve_aliases(), _check_required_columns().
- It standardizes column names, filters DTE ranges, and produces processed datasets for full-chain and signal-data use.

### strategy/signal.py
- Contains an older, more feature-rich signal engine for IV vs RV signals.
- Classes/functions: volSignalEngine, run_signal_pipeline().
- It computes realized vol, forward RV, spread features, skew, term-structure, liquidity, regime, and outputs a composite signal score.

### strategy/rules.py
- Applies execution rules to the raw signal stream.
- Classes/functions: RuleConfig, VolTradeRules, run_rules().
- It enforces liquidity gating, lagging, cooldowns, max holding periods, and position flip restrictions.

### strategy/sizing.py
- Determines trade quantity based on risk target.
- Classes/functions: SizerConfig, VolSizer, calculate_quantity().
- Supports vega-based sizing, volatility-target sizing, and kelly-style sizing with notional caps.

### strategy/position.py
- Defines the straddle position object and strike selection logic.
- Classes/functions: StraddlePosition, PositionManager, create_straddle(), select_strike().
- It chooses a strike (ATM/near-ATM using delta or absolute strike distance) and tracks Greeks and PnL attribution.

### execution/delta_hedge.py
- Implements the hedge engine for delta-neutralization.
- Classes/functions: HedgeConfig, HedgeAction, DeltaHedgeEngine, calculate_hedge_action().
- It computes share trades to offset portfolio delta and estimates hedge cost/slippage.

### execution/transactions_costs.py
- Models option and stock trading costs.
- Classes/functions: CostConfig, TransactionalCostModel.
- It estimates commissions, bid/ask spreads, slippage, and SEC/TAF-like fees.

### backtest/engine.py
- The core event-driven backtest engine.
- Classes/functions: BacktestConfig, VolBacktest, run(), _select_entry_row(), _lookup().
- It iterates through daily signals, opens/closes positions, marks to market, applies hedges, and outputs a results DataFrame.

### backtest/portfolio.py
- Maintains portfolio cash, position state, hedge shares, and cumulative PnL buckets.
- Classes/functions: Portfolio, open_position(), close_position(), apply_hedge(), mark_to_market().
- It tracks option PnL, hedge PnL, costs, and Greek-attribution buckets for the strategy.

### backtest/performance.py
- Computes risk-adjusted performance and trade statistics.
- Functions: compute_performance(), _trade_statistics().
- It returns metrics like total return, CAGR, Sharpe, Sortino, Calmar, drawdown, win rate, and profit factor.

### bs/pricing.py and bs/greeks.py
- Implement Black-Scholes pricing and Greeks.
- Functions: bs_price(), delta(), gamma(), vega().
- These are foundational for option valuation and Greek-based attribution.

### bs/implied_vol.py
- Solves implied vol from market prices using Newton and bisection methods.
- Functions: implied_vol_newton(), implied_vol_bisection(), implied_vol().
- It is used to estimate the IV of each option contract.

### data/loaders.py
- Downloads option-chain data from Yahoo Finance or CBOE.
- Functions: load_option_chain_yahoo(), load_option_chain_cboe().
- It fetches options chains and prepares a simplified dataframe for IV calculation.

### scripts/build_iv_surface.py
- Experimental script to build and plot an IV surface from option data.
- Functions: run_live_loop().
- It is not wired into the main backtest path.

---

## 4. DEPENDENCIES

### Python dependencies (from requirements.txt)
Main libraries:
- numpy
- pandas
- scipy
- yfinance
- requests
- pytz
- plotly
- matplotlib

The project also uses:
- pyarrow/parquet support (implicitly through pandas and parquet IO)
- standard-library dataclasses, logging, typing

### Internal/custom modules used heavily
- data.data_pipeline
- strategy.signal
- strategy.rules
- strategy.sizing
- strategy.position
- execution.delta_hedge
- execution.transactions_costs
- backtest.engine
- backtest.portfolio
- backtest.performance
- bs.pricing / bs.greeks / bs.implied_vol
- vol.metrics / vol.realized_vol

### Notes
- There is no package.json or frontend stack in this workspace.
- The project is Python-first and mostly script-driven.

---

## 5. DATA FLOW

### Source of data
- Primary source appears to be a local parquet file: data/SPY_ALL_YEARS_MASTER.parquet
- There is also support for downloading live option-chain data from Yahoo Finance or CBOE via data/loaders.py

### Processing path
1. Raw option-chain data is loaded from parquet/CSV.
2. Column names are normalized and aliases are resolved.
3. DTE and strike filters are applied.
4. Two datasets are built:
   - full_chain: broad option chain used in backtest and for position hold continuity
   - signal_data: narrower signal dataset for signal-feature computation
5. Signal engine computes IV/RV spread, term structure, skew, regime, liquidity.
6. Trade rules filter/convert signals into executable positions.
7. Sizing chooses quantity.
8. Backtest engine opens/closes positions, applies hedge, computes PnL and metrics.

### Output destinations
- Processed parquet files in data/processed/
- Performance chart in logs/backtest_results.png
- Console tear sheet in terminal
- Optional plots from visualization utilities

---

## 6. CORE LOGIC / STRATEGY DETAILS

### Core strategy idea
The system is built around a volatility mean-reversion thesis:
- If implied volatility is rich relative to realized volatility, the strategy may short volatility exposure (or fade IV spikes).
- If IV is cheap relative to RV, the strategy may go long volatility exposure.

### Signal logic
The main signal pipeline uses feature components such as:
- IV-RV spread
- spread z-score / percentile
- regime score
- skew signal
- term-structure slope
- liquidity score
- composite “out” score

Key signal-related functions:
- run_signal_pipeline(data, config=None)
- volSignalEngine.compute_features()
- _compute_term_structure_score()
- _compute_skew()
- _compute_contributions()
- _generate_signal()

### Trade logic
- Signals are converted to position states (+1/-1/0) with entry/exit logic.
- Rules enforce lag, liquidity, cooldowns, holding period caps, and no-flip behavior.

### Risk / execution logic
- Position sizing uses vega or risk-based sizing.
- Delta hedging offsets directional risk.
- Transaction costs are modeled explicitly for options and equity hedges.

### Backtest logic
- Daily event-driven simulation
- Entry at signal flags
- Exit on signal exit, stop-loss/profit-take conditions, or max holding limit
- Mark-to-market each day and update PnL / Greeks

---

## 7. CONFIG & ENV

### Config style
The project mostly uses hard-coded Python configuration rather than YAML/JSON/env files.

Examples of hard-coded settings:
- Raw path: data/SPY_ALL_YEARS_MASTER.parquet
- Processed output paths in scripts/run_backtest.py
- Signal config in strategy/signal.py (window, entry_z, exit_z, weights)
- Rule config in strategy/rules.py (execution lag, liquidity threshold, max holding days)
- Sizing config in strategy/sizing.py
- Hedge config in execution/delta_hedge.py
- Backtest config in backtest/engine.py

### Environment notes
- There is a local virtualenv under quant/
- No broker API credentials or secrets are present in the visible tree
- No obvious .env or config.yaml files were found in the root

---

## 8. CURRENT STATE / TODOs

### What appears to be working
- The main backtest runner executes successfully with the project venv.
- Data pipeline and processed parquet caching work.
- Basic performance reporting and chart generation work.

### Known issues / observations
- The backtest output shows significant data-gap warnings and a large residual in Greek attribution.
- The engine currently logs data-gap warnings on exit/lookup events.
- There are explicit comments in the code around “FIX” and “patch” logic for data gaps and IV carry-forward.
- The project contains several experimental or legacy modules:
  - singal.py (typo in filename, likely older signal logic)
  - vol/metrcis(gpt).py (experimental metrics module)
  - scripts/build_iv_surface.py / live_surface.py (exploratory IV surface work)
- There is a TODO-style integration note in singal.py indicating future work to integrate with vol/iv_surface.py.

### Practical interpretation
The repo is functional as a research prototype, but it is still in a debugging / refinement phase rather than a polished production platform.

---

## 9. ENTRY POINT

### Primary entry point
The main runnable script is:

```python
# scripts/run_backtest.py
if __name__ == "__main__":
    main()
```

### Command to run
From the project root:

```bash
./quant/bin/python -m scripts.run_backtest
```

Alternative (if the system Python has the required deps installed):

```bash
python -m scripts.run_backtest
```

---

## 10. TERMINAL RUN RESULTS (verified)

Command run:

```bash
./quant/bin/python -m scripts.run_backtest
```

Observed output summary:
- Backtest initialized and cached parquet data loaded successfully
- Full chain rows: 3,996,866
- Signal data rows: 449,025
- Strategy ran through signal pipeline, rules, sizing, hedging, and backtest engine
- Final performance summary:
  - Total Return: 0.11%
  - CAGR: 0.01%
  - Ann. Volatility: 0.55%
  - Sharpe Ratio: -8.84
  - Sortino Ratio: -2.14
  - Max Drawdown: -1.78%
  - Total Option PnL: $5,267
  - Total Hedge PnL: $4,689
  - Total Frictions: $5,815
  - Num Trades: 22
  - Win Rate: 59.1%
  - Profit Factor: 1.31
  - Avg Holding Days: 6.5
- The run also reported data-gap warnings and a large Greek attribution residual (~190.1%), indicating that there are still robustness issues in the data/lookup pipeline.

---

## 11. QUICK RECOMMENDATION FOR NEXT AI AGENT

Best next steps for a follow-up agent:
1. Investigate the data-gap and lookup issues in the backtest engine
2. Verify whether the IV carry-forward logic is sufficient for Greek attribution
3. Clean up or prune experimental modules (singal.py, metrcis(gpt).py)
4. Add configuration files or a cleaner settings layer
5. Consider making the strategy more production-like with live data integration and broker adapters
