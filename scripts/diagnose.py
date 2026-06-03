"""
diagnose_gaps.py
----------------
Run this to figure out WHY the engine reports data gaps even though
the OptionsDX source data should be complete.

Usage:
    python diagnose_gaps.py

Checks three things in order:
  1. Do the gap (date, strike) rows actually exist in full_chain.parquet?
  2. If yes → the bug is in the engine's index lookup (dtype / tz mismatch).
  3. Print the exact dtypes and sample index values so the engine fix is obvious.
"""

import pandas as pd
import numpy as np

FULL_CHAIN_PATH  = "data/processed/full_chain.parquet"
SIGNAL_DATA_PATH = "data/processed/signal_data.parquet"

# Every (date, strike) pair the engine flagged as a gap
GAP_EVENTS = [
    ("2011-05-10", 134.0),
    ("2014-06-03", 192.0),
    ("2015-08-25", 198.0),
    ("2016-08-16", 218.5),
    ("2016-09-06", 217.5),
    ("2016-11-07", 211.0),
    ("2017-04-24", 233.5),
    ("2018-01-22", 279.5),
    ("2018-10-15", 277.5),
    ("2019-05-10", 288.0),
    ("2021-01-04", 372.0),
    ("2023-11-07", 439.0),
    ("2023-11-20", 451.0),
    ("2023-12-01", 461.0),
]

SEP = "─" * 60


def check_full_chain():
    print(f"\n{SEP}")
    print("STEP 1 — Load full_chain and inspect dtypes")
    print(SEP)

    fc = pd.read_parquet(FULL_CHAIN_PATH)
    fc["quote_date"]  = pd.to_datetime(fc["quote_date"])
    fc["expire_date"] = pd.to_datetime(fc["expire_date"])

    print(f"Rows          : {len(fc):,}")
    print(f"Dates         : {fc['quote_date'].nunique()}")
    print(f"strike dtype  : {fc['strike'].dtype}")
    print(f"quote_date tz : {fc['quote_date'].dt.tz}")
    print(f"expire_date tz: {fc['expire_date'].dt.tz}")
    print(f"\nSample strikes: {sorted(fc['strike'].unique()[:8])}")

    print(f"\n{SEP}")
    print("STEP 2 — Check each gap event in full_chain")
    print(SEP)

    found   = []
    missing = []

    for date_str, strike in GAP_EVENTS:
        target_date = pd.Timestamp(date_str)
        mask = (
            (fc["quote_date"].dt.normalize() == target_date) &
            (fc["strike"] == strike)
        )
        rows = fc[mask]
        if len(rows) > 0:
            expire_dates = rows["expire_date"].dt.strftime("%Y-%m-%d").unique()
            found.append((date_str, strike, len(rows), list(expire_dates[:4])))
        else:
            # Try fuzzy match — maybe the strike is stored as 218.50000001 etc.
            near = fc[
                (fc["quote_date"].dt.normalize() == target_date) &
                (np.abs(fc["strike"] - strike) < 0.01)
            ]
            missing.append((date_str, strike, len(near)))

    print(f"\n✓ FOUND in full_chain ({len(found)}/{len(GAP_EVENTS)} gap events):")
    for date_str, strike, n, expiries in found:
        print(f"  {date_str}  strike={strike:<7}  {n:>3} rows  expiries={expiries}")

    print(f"\n✗ MISSING from full_chain ({len(missing)}/{len(GAP_EVENTS)} gap events):")
    for date_str, strike, fuzzy_n in missing:
        note = f"(fuzzy match found {fuzzy_n} rows — float precision issue?)" if fuzzy_n > 0 else "(genuinely absent)"
        print(f"  {date_str}  strike={strike:<7}  {note}")

    return fc, found, missing


def check_signal_dtypes(found_gaps):
    """
    If data IS in full_chain, the bug is in how signals pass the key
    to the engine.  Check the signal/executable signal dtypes.
    """
    if not found_gaps:
        return

    print(f"\n{SEP}")
    print("STEP 3 — Check signal_data dtypes (engine lookup key source)")
    print(SEP)

    sd = pd.read_parquet(SIGNAL_DATA_PATH)
    sd["quote_date"]  = pd.to_datetime(sd["quote_date"])
    sd["expire_date"] = pd.to_datetime(sd["expire_date"])

    print(f"strike dtype  : {sd['strike'].dtype}   ← should match full_chain")
    print(f"quote_date tz : {sd['quote_date'].dt.tz}")
    print(f"expire_date tz: {sd['expire_date'].dt.tz}")

    # Check if any signal strikes are float32 (would fail a float64 index lookup)
    if sd["strike"].dtype == np.float32:
        print("\n⚠  strike is float32 in signal_data but float64 in full_chain.")
        print("   The engine's MultiIndex.loc(strike) call will silently miss every row.")
        print("   Fix: add sd['strike'] = sd['strike'].astype(np.float64) in load_signal_data()")
    else:
        print("\n✓ strike dtype matches — dtype mismatch is NOT the cause.")
        print("  Next place to check: how the engine builds its MultiIndex key.")
        print("  Look for this pattern in backtest/engine.py:")
        print("    fc.set_index(['quote_date','strike','expire_date'])")
        print("  and verify the key passed to .loc[] uses the exact same dtypes.")


def check_engine_index_hint(fc):
    """
    Show what a correctly-built MultiIndex lookup should look like,
    using a real found gap as a worked example.
    """
    print(f"\n{SEP}")
    print("STEP 4 — Correct MultiIndex lookup pattern for engine.py")
    print(SEP)

    # Use first gap as example
    date_str, strike = GAP_EVENTS[0]
    target_date = pd.Timestamp(date_str)
    sample = fc[
        (fc["quote_date"].dt.normalize() == target_date) &
        (fc["strike"] == strike)
    ]

    if sample.empty:
        print(f"  (No sample row available for {date_str} / {strike})")
        return

    row = sample.iloc[0]
    qd  = row["quote_date"]
    ed  = row["expire_date"]
    s   = row["strike"]

    print(f"\nExample key for {date_str} / strike {strike}:")
    print(f"  quote_date  : {qd!r}  (type: {type(qd).__name__}, tz: {qd.tz})")
    print(f"  expire_date : {ed!r}  (type: {type(ed).__name__}, tz: {ed.tz})")
    print(f"  strike      : {s!r}  (dtype: {type(s).__name__})")

    print("""
Recommended engine lookup (add to backtest/engine.py):

    # Build once at engine startup
    _fc_idx = full_chain.set_index(['quote_date', 'expire_date', 'strike']).sort_index()

    # Per-day lookup
    def get_contract(quote_date, expire_date, strike):
        key = (
            pd.Timestamp(quote_date),           # normalize tz-naive
            pd.Timestamp(expire_date),
            float(strike),                       # always float64
        )
        try:
            return _fc_idx.loc[key]
        except KeyError:
            return None

The three most common reasons .loc[] silently misses:
  1. strike is float32 in the key, float64 in the index
  2. quote_date has a timezone in one place but not the other
  3. expire_date was stored as date-only (00:00:00) but the key has a time component
""")


def main():
    print("=" * 60)
    print("  DATA GAP DIAGNOSTIC")
    print("=" * 60)

    fc, found, missing = check_full_chain()
    check_signal_dtypes(found)
    check_engine_index_hint(fc)

    print(SEP)
    print("SUMMARY")
    print(SEP)
    if missing and not found:
        print("→ All gaps are GENUINELY missing from full_chain.parquet.")
        print("  This points to a filtering issue in build_full_chain().")
        print("  Check FULL_CHAIN_DTE_MIN — lower it to 0 to capture expiry-date rows.")
    elif found and not missing:
        print("→ ALL gap strikes ARE in full_chain. The data is fine.")
        print("  The engine's index lookup is failing on dtype or timezone mismatch.")
        print("  Apply the lookup pattern shown in STEP 4 to engine.py.")
    else:
        print(f"→ Mixed result: {len(found)} found, {len(missing)} missing.")
        print("  Fix the missing rows first (DTE filter), then fix the lookup.")
    print()


if __name__ == "__main__":
    main()