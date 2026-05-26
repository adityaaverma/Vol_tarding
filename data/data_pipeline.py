import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
 
logger = logging.getLogger(__name__)
 
# ── DTE windows ──────────────────────────────────────────────────────────────
FULL_CHAIN_DTE_MIN  = 5     # exclude same-day expiry
FULL_CHAIN_DTE_MAX  = 120   # exclude very long-dated illiquid LEAPS
 
SIGNAL_DTE_MIN      = 25    # near enough to have meaningful theta
SIGNAL_DTE_MAX      = 55    # 25-55 DTE is the sweet spot for ATM IV signal

 
# ── moneyness filter for signal slice ────────────────────────────────────────
# abs(log(S/K)) < 0.05  →  roughly ±5% moneyness

SIGNAL_MONEYNESS_CUTOFF = 0.05

def _parse_contract_size(val) -> float:
    """'10 x 100' → 1000.0.  Anything else → NaN."""
    try:
        parts = str(val).split(" x ")
        return float(parts[0]) * float(parts[1])
    except Exception:
        return np.nan
 
 
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        c.replace("[", "").replace("]", "").replace(" ", "").strip().lower()
        for c in df.columns
    ]
    return df


def _safe_numeric_cast(df: pd.DataFrame, skip_cols: list) -> pd.DataFrame:
    """
    Convert every non-skip column to numeric (float64).
    Blank strings become NaN instead of causing ValueErrors.
    Does NOT downcast to float32 — keeps full precision for MultiIndex lookups.
    """
    for col in df.columns:
        if col in skip_cols:
            continue
        # Use pd.to_numeric with errors='coerce' to convert all non-numeric values to NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_raw(path: str) -> pd.DataFrame:
    """
    Load the raw Kaggle SPY options CSV and apply minimal, safe cleaning.
    Returns float64 throughout — no float32 downcasting.
    """
    logger.info(f"Loading raw data from {path} …")
    df = pd.read_csv(path, low_memory=False)
    df = _clean_column_names(df)
 
    # ── contract size ─────────────────────────────────────────────────────────
    for col in ("c_size", "p_size"):
        if col in df.columns:
            df[col] = df[col].apply(_parse_contract_size)
 
    # ── datetime columns ──────────────────────────────────────────────────────
    date_cols = [c for c in df.columns if "date" in c or "readtime" in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
 
    # ── numeric columns (everything else, except unix timestamps which are int)
    skip = set(date_cols) | {"quote_unixtime", "expire_unix"}
    df = _safe_numeric_cast(df, skip_cols=list(skip))
 
    # ── derived columns ───────────────────────────────────────────────────────
    df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
    df["moneyness"]     = np.log(df["underlying_last"] / df["strike"])
    df["abs_moneyness"] = df["moneyness"].abs()
 
    logger.info(f"Raw data loaded: {len(df):,} rows, {df['quote_date'].nunique()} dates.")
    return df


def build_full_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full options chain for the backtest engine (options_df).
 
    Filters:
      • Both bid AND ask > 0  (contract is tradeable)
      • DTE in [FULL_CHAIN_DTE_MIN, FULL_CHAIN_DTE_MAX]
 
    Does NOT filter on IV — a contract can be tradeable even if the IV
    solver returned NaN (deep ITM, wide spread). The backtest can still
    price it via bid/ask.
 
    Keeps strike as float64 so MultiIndex.loc() lookups work correctly.
    """
    mask = (
        (df["c_bid"] > 0) & (df["c_ask"] > 0) &
        (df["p_bid"] > 0) & (df["p_ask"] > 0) &
        (df["dte"] >= FULL_CHAIN_DTE_MIN) &
        (df["dte"] <= FULL_CHAIN_DTE_MAX)
    )
    chain = df[mask].copy()
 
    # Guarantee float64 on the three MultiIndex keys
    chain["strike"]      = chain["strike"].astype(np.float64)
    chain["quote_date"]  = pd.to_datetime(chain["quote_date"])
    chain["expire_date"] = pd.to_datetime(chain["expire_date"])
 
    logger.info(
        f"Full chain: {len(chain):,} rows "
        f"({chain['quote_date'].nunique()} dates, "
        f"{chain['strike'].nunique()} unique strikes)."
    )
    return chain.reset_index(drop=True)

def build_signal_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, liquid ATM slice for signal feature computation.
 
    Filters:
      • DTE in [SIGNAL_DTE_MIN, SIGNAL_DTE_MAX]
      • abs_moneyness < SIGNAL_MONEYNESS_CUTOFF  (near-ATM only)
      • Both c_iv AND p_iv present                (IV signal needs both)
      • Both bid/ask > 0
 
    This is what goes into volSignalEngine.compute_features().
    """
    mask = (
        (df["dte"] >= SIGNAL_DTE_MIN) &
        (df["dte"] <= SIGNAL_DTE_MAX) &
        (df["abs_moneyness"] < SIGNAL_MONEYNESS_CUTOFF) &
        (df["c_iv"].notna()) &
        (df["p_iv"].notna()) &
        (df["c_bid"] > 0) & (df["c_ask"] > 0) &
        (df["p_bid"] > 0) & (df["p_ask"] > 0)
    )
    signal = df[mask].copy()
 
    logger.info(
        f"Signal slice: {len(signal):,} rows "
        f"({signal['quote_date'].nunique()} dates)."
    )
    return signal.reset_index(drop=True)


def _validate(full_chain: pd.DataFrame, signal_data: pd.DataFrame) -> None:
    """Sanity checks that catch the original bugs at pipeline time."""
 
    # Check 1: strike dtype
    assert full_chain["strike"].dtype == np.float64, \
        "Strike must be float64 for MultiIndex lookups to work"
 
    # Check 2: signal dates are a subset of full_chain dates
    signal_dates = set(signal_data["quote_date"].dt.date)
    chain_dates  = set(full_chain["quote_date"].dt.date)
    orphan_dates = signal_dates - chain_dates
    if orphan_dates:
        logger.warning(
            f"{len(orphan_dates)} signal dates have no full_chain rows — "
            "entries on those dates will fail."
        )
 
    # Check 3: coverage — for each (date, strike, expiry) in a signal entry,
    #           check that the full_chain has rows for the next 30 days
    # (lightweight spot-check on first 5 signal dates)
    sample_dates = sorted(signal_dates)[:5]
    for d in sample_dates:
        day_chain = full_chain[full_chain["quote_date"].dt.date == d]
        n = len(day_chain)
        if n == 0:
            logger.warning(f"No full_chain rows for signal date {d}")
        else:
            logger.debug(f"  {d}: {n} contracts in full_chain ✓")
 
    # Check 4: DTE coverage — full_chain should span SIGNAL_DTE_MIN to 0
    min_dte = full_chain["dte"].min()
    if min_dte > SIGNAL_DTE_MIN:
        logger.warning(
            f"full_chain min DTE is {min_dte}, but signal entries can have "
            f"DTE as low as {SIGNAL_DTE_MIN}. You will get data gaps as "
            f"positions age past {min_dte} DTE."
        )
 
    logger.info("Validation complete.")


def load_full_chain(path: str = "data/processed/full_chain.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["quote_date"]  = pd.to_datetime(df["quote_date"])
    df["expire_date"] = pd.to_datetime(df["expire_date"])
    df["strike"]      = df["strike"].astype(np.float64)   # guarantee float64
    return df
 
 
def load_signal_data(path: str = "data/processed/signal_data.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["quote_date"]  = pd.to_datetime(df["quote_date"])
    df["expire_date"] = pd.to_datetime(df["expire_date"])
    return df


def run_pipeline(
    raw_path: str,
    output_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline.  Returns (full_chain, signal_data) and
    writes both to parquet (preferred) or CSV.
 
    Usage
    -----
    full_chain, signal_data = run_pipeline("data/raw/spy_2020_2022.csv")
 
    Then in your backtest runner:
        signals_df = run_signal_pipeline(signal_data)   # signal.py
        signals_df = run_rules(signals_df)              # rules.py
        backtest   = VolBacktest(signals_df, full_chain, ...)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    raw        = load_raw(raw_path)
    full_chain = build_full_chain(raw)
    signal_data = build_signal_data(raw)
 
    # Prefer parquet (10× faster to load, preserves dtypes exactly)
    try:
        full_chain.to_parquet(out / "full_chain.parquet", index=False)
        signal_data.to_parquet(out / "signal_data.parquet", index=False)
        logger.info(f"Saved parquet files to {out}/")
    except ImportError:
        full_chain.to_csv(out / "full_chain.csv", index=False)
        signal_data.to_csv(out / "signal_data.csv", index=False)
        logger.info(f"pyarrow not installed — saved CSV files to {out}/")
 
    _validate(full_chain, signal_data)
    return full_chain, signal_data

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/spy_2020_2022.csv"
    run_pipeline(raw_path)