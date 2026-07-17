#!/usr/bin/env python3
"""
Comprehensive column name consistency checker.
Scans all Python files for column references and verifies they match canonical names.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# Canonical column names (lowercase with underscores)
CANONICAL_COLS = {
    # underlying / spot
    "underlying_last",
    # dates
    "quote_date", "expire_date",
    # expiration
    "dte",
    # strike
    "strike",
    # call Greeks and prices
    "c_bid", "c_ask", "c_iv", "c_delta", "c_gamma", "c_vega", "c_theta",
    # put Greeks and prices
    "p_bid", "p_ask", "p_iv", "p_delta", "p_gamma", "p_vega", "p_theta",
    # computed columns
    "moneyness", "abs_moneyness", "iv", "rv", "fwd_rv",
    "spread", "fwd_spread", "spread_z", "spread_percentile",
    "iv_change_1d", "iv_change_5d", "rv_change_5d",
    "liquidity_score", "skew", "term_slope", "term_slope_raw",
    "regime_score", "skew_z", "call_put_iv_gap", "returns",
    # backtest output
    "nav", "cash", "shares", "date", "position", "position_prev",
    "has_position", "option_pnl_change", "cumulative_option_pnl",
    "cumulative_hedge_pnl", "cumulative_costs", "cumulative_delta_pnl",
    "cumulative_gamma_pnl", "cumulative_vega_pnl", "cumulative_theta_pnl",
    "signal", "drawdown",
    # other
    "expiry", "ticker",
}

# Columns that are acceptable in external data sources (loaders, live data, etc.)
EXTERNAL_COLS = {
    "K", "P", "T", "option_type", "spot", "iv_yf",  # Yahoo Finance loader
    "bid", "ask", "strike", "impliedVolatility", "openInterest", "lastPrice",  # YF raw
    "reported_nav", "calculated_nav", "atm_iv",  # computed/output
}

def find_column_references(file_path):
    """Extract all column references from Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find patterns like df['col_name'] or df["col_name"]
    pattern = r"[\w]+\[(['\"])([A-Za-z_][A-Za-z0-9_]*)\1\]"
    matches = re.findall(pattern, content)
    
    cols = defaultdict(list)
    for _, col_name in matches:
        cols[col_name].append(file_path)
    
    return cols

def check_project_consistency():
    """Check all Python files for column name consistency."""
    project_root = Path(__file__).parent
    py_files = list(project_root.rglob('*.py'))
    
    found_issues = False
    suspicious_cols = defaultdict(list)
    
    print("=" * 70)
    print("COLUMN NAME CONSISTENCY CHECK")
    print("=" * 70)
    
    for py_file in py_files:
        # Skip venv and cache
        if any(part in py_file.parts for part in ['quant', '__pycache__', '.pytest_cache', '.venv', 'site-packages']):
            continue
        
        cols = find_column_references(py_file)
        
        for col_name, files in cols.items():
            # Check if it's an uppercase version of a canonical column
            lower_col = col_name.lower()
            
            # Skip certain patterns
            if col_name in CANONICAL_COLS or col_name in EXTERNAL_COLS:
                continue
            
            # Check for uppercase versions of canonical columns
            if lower_col in CANONICAL_COLS and col_name != lower_col:
                found_issues = True
                suspicious_cols[col_name].append((lower_col, str(py_file)))
                print(f"\n⚠️  MISMATCH FOUND:")
                print(f"   Column: '{col_name}' should be '{lower_col}'")
                print(f"   File: {py_file.relative_to(project_root)}")
            
            # Check for mixed case or unusual patterns
            if not col_name.islower() and '_' not in col_name and col_name not in EXTERNAL_COLS:
                if re.match(r'^[A-Z][A-Za-z]+$', col_name):  # like "Signal" or "Date"
                    suspicious_cols[col_name].append((f"possibly '{col_name.lower()}'", str(py_file)))
                    print(f"\n⚠️  SUSPICIOUS:")
                    print(f"   Column: '{col_name}' (in {py_file.relative_to(project_root)})")
                    print(f"   Suggestion: Consider if this should be '{col_name.lower()}'")
    
    # Summary
    print("\n" + "=" * 70)
    if found_issues:
        print("❌ INCONSISTENCIES DETECTED - Please fix the above issues")
        return False
    else:
        print("✅ All column references are consistent!")
        return True

if __name__ == "__main__":
    success = check_project_consistency()
    sys.exit(0 if success else 1)
