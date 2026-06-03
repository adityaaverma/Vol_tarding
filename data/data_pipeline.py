import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# ── DTE windows ───────────────────────────────────────────────────────────────
FULL_CHAIN_DTE_MIN  = 5
FULL_CHAIN_DTE_MAX  = 120
SIGNAL_DTE_MIN      = 25
SIGNAL_DTE_MAX      = 55
SIGNAL_MONEYNESS_CUTOFF = 0.05

# ── FIX 1: Column alias map ───────────────────────────────────────────────────
# Different data vendors use different names for the same field.
# Map every known alias → the canonical name this codebase uses.
# Add more aliases here if your parquet uses yet another convention.
COLUMN_ALIASES: dict[str, str] = {
    # underlying price
    "underlying_price":  "underlying_last",
    "undprice":          "underlying_last",
    "spot":              "underlying_last",
    "spx":               "underlying_last",
    "close":             "underlying_last",
    # quote date
    "quotedate":         "quote_date",
    "tradedate":         "quote_date",
    "date":              "quote_date",
    # expire date
    "expiration":        "expire_date",
    "expiry":            "expire_date",
    "expdate":           "expire_date",
    "expirationdate":    "expire_date",
    # days to expiration
    "daystoexpiration":  "dte",
    "days_to_exp":       "dte",
    "days":              "dte",
    # strike
    "strikeprice":       "strike",
    "strike_price":      "strike",
    # call greeks / prices
    "call_bid":          "c_bid",
    "call_ask":          "c_ask",
    "call_iv":           "c_iv",
    "call_delta":        "c_delta",
    "call_gamma":        "c_gamma",
    "call_vega":         "c_vega",
    "call_theta":        "c_theta",
    # put greeks / prices
    "put_bid":           "p_bid",
    "put_ask":           "p_ask",
    "put_iv":            "p_iv",
    "put_delta":         "p_delta",
    "put_gamma":         "p_gamma",
    "put_vega":          "p_vega",
    "put_theta":         "p_theta",
}

# Columns that MUST exist after alias resolution for the pipeline to proceed.
REQUIRED_COLS = [
    "quote_date", "expire_date", "strike",
    "underlying_last",
    "c_bid", "c_ask",
    "p_bid", "p_ask",
]


def inspect_schema(path: str) -> None:
    """
    FIX: Diagnostic utility — call this first whenever the pipeline crashes.
    Prints every column name and dtype without loading the full dataset.
    Usage: python -c "from data.data_pipeline import inspect_schema; inspect_schema('data/SPY_ALL_YEARS_MASTER.parquet')"
    """
    logger.info(f"=== Schema inspection: {path} ===")
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        print(f"\nParquet schema ({len(schema)} fields):")
        for field in schema:
            print(f"  {field.name!r:45s} {str(field.type)}")
    else:
        sample = pd.read_csv(path, nrows=5)
        print(f"\nCSV columns ({len(sample.columns)} fields):")
        for col, dtype in sample.dtypes.items():
            print(f"  {col!r:45s} {dtype}")
    print()


def _parse_contract_size(val) -> float:
    """'10 x 100' → 1000.0.  Anything else → NaN."""
    try:
        parts = str(val).split(" x ")
        return float(parts[0]) * float(parts[1])
    except Exception:
        return np.nan


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip brackets/spaces and lowercase — handles OptionsDX raw format."""
    df.columns = [
        c.replace("[", "").replace("]", "").replace(" ", "").strip().lower()
        for c in df.columns
    ]
    return df


def _resolve_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 1: Rename any aliased column names to their canonical equivalents.
    Silently skips aliases that don't appear in the dataframe.
    """
    rename_map = {
        alias: canonical
        for alias, canonical in COLUMN_ALIASES.items()
        if alias in df.columns and canonical not in df.columns
    }
    if rename_map:
        logger.info(f"Resolving column aliases: {rename_map}")
        df = df.rename(columns=rename_map)
    return df


def _check_required_columns(df: pd.DataFrame) -> None:
    """
    FIX 2: Fail fast with a clear message listing missing columns and all
    available columns, instead of crashing with an opaque KeyError later.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        available = sorted(df.columns.tolist())
        raise KeyError(
            f"\n\n{'='*60}\n"
            f"MISSING REQUIRED COLUMNS: {missing}\n\n"
            f"Available columns in your file ({len(available)}):\n"
            f"  {available}\n\n"
            f"Add the missing column names to COLUMN_ALIASES in data_pipeline.py\n"
            f"or run inspect_schema(path) to see your file's exact column names.\n"
            f"{'='*60}"
        )


def _safe_numeric_cast(df: pd.DataFrame, skip_cols: list) -> pd.DataFrame:
    """
    Convert every non-skip column to numeric (float64).
    Blank strings and other non-numeric values become NaN.
    Does NOT downcast — keeps float64 for MultiIndex lookups.
    """
    for col in df.columns:
        if col in skip_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _compute_dte_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 3: Some vendors don't include a pre-computed DTE column.
    Derive it from expire_date and quote_date when missing or all-NaN.
    """
    if "dte" not in df.columns or df["dte"].isna().all():
        logger.info("'dte' column missing or all-NaN — computing from expire_date - quote_date.")
        df["dte"] = (
            pd.to_datetime(df["expire_date"]) - pd.to_datetime(df["quote_date"])
        ).dt.days
    return df


# ── Memory optimisation helpers ───────────────────────────────────────────────

# Only read these columns from the parquet — skips 6 unused columns and
# shaves ~15% off peak memory before any filtering is done.
_PARQUET_COLS = [
    "QUOTE_UNIXTIME", "QUOTE_DATE", "UNDERLYING_LAST",
    "EXPIRE_DATE", "EXPIRE_UNIX", "DTE",
    "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO",
    "C_IV", "C_LAST", "C_SIZE", "C_BID", "C_ASK",
    "STRIKE",
    "P_BID", "P_ASK", "P_SIZE", "P_LAST",
    "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO",
    "P_IV",
]

# Columns that MUST stay float64 (MultiIndex keys + unix timestamps)
_KEEP_FLOAT64 = {"strike", "quote_unixtime", "expire_unix"}


def _log_memory(df: pd.DataFrame, label: str) -> None:
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info(f"  RAM [{label}]: {mb:.0f} MB  ({len(df):,} rows)")


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    float64 → float32 for every non-key numeric column.
    Saves ~40% RAM. strike stays float64 for MultiIndex lookups.
    """
    for col in df.select_dtypes(include="float64").columns:
        if col not in _KEEP_FLOAT64:
            df[col] = df[col].astype(np.float32)
    return df


def _load_parquet_batched(path: str, batch_size: int = 500_000) -> pd.DataFrame:
    """
    Read parquet in 500 k-row batches.  Applies DTE filter and float32
    downcast *per batch* so peak RAM is (batch × 27 cols) not (full file).

    Strategy
    --------
    1. Read batch                       →  ~130 MB  (500 k rows × 27 cols × float64)
    2. Lowercase column names
    3. Cast DTE to numeric; filter DTE 5-120  →  drops ~40-60% of rows
    4. Downcast survivors to float32    →  halves memory for the kept rows
    5. Append filtered batch to list
    Peak memory ≈ one raw batch + growing filtered list, never the full file.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    file_cols = {f.name for f in pf.schema_arrow}

    # Only request columns that actually exist in this file
    read_cols = [c for c in _PARQUET_COLS if c in file_cols]
    missing   = set(_PARQUET_COLS) - file_cols
    if missing:
        logger.warning(f"Columns not found in parquet (skipped): {sorted(missing)}")

    chunks: list[pd.DataFrame] = []
    total_in = 0
    total_out = 0

    for batch_num, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=read_cols), 1):
        df_b = batch.to_pandas()
        total_in += len(df_b)

        # Normalise names in-place (no allocation)
        df_b = _clean_column_names(df_b)
        df_b = _resolve_aliases(df_b)

        # DTE filter — do this as early as possible to shrink the batch
        if "dte" in df_b.columns:
            df_b["dte"] = pd.to_numeric(df_b["dte"], errors="coerce")
            df_b = df_b[
                (df_b["dte"] >= FULL_CHAIN_DTE_MIN) &
                (df_b["dte"] <= FULL_CHAIN_DTE_MAX)
            ]

        if df_b.empty:
            logger.info(f"  Batch {batch_num}: {batch_size:,} rows → 0 kept (all outside DTE range)")
            continue

        # Downcast floats to float32 on the filtered slice only
        df_b = _downcast_floats(df_b)

        total_out += len(df_b)
        chunks.append(df_b)
        logger.info(
            f"  Batch {batch_num}: {total_in:,} rows read so far, "
            f"{total_out:,} kept ({100 * total_out / total_in:.1f}%)"
        )

    if not chunks:
        raise ValueError(
            f"No rows survived the DTE filter [{FULL_CHAIN_DTE_MIN}, {FULL_CHAIN_DTE_MAX}]. "
            "Check that your parquet file contains valid DTE values."
        )

    df = pd.concat(chunks, ignore_index=True)
    _log_memory(df, "after batch load + DTE filter")
    return df


def load_raw(path: str) -> pd.DataFrame:
    """
    Load the raw SPY options data (parquet or CSV) and apply minimal, safe cleaning.

    Key changes vs the original:
      • inspect_schema()-guided alias resolution so column mismatches fail clearly
      • DTE fallback computed from dates if the column is absent
      • Guard against zero/negative strikes before log() to prevent -inf moneyness
      • c_size / p_size parsing is skipped gracefully when the columns don't exist
    """
    logger.info(f"Loading raw data from {path} …")

    # ── read ──────────────────────────────────────────────────────────────────
    try:
        if path.endswith(".parquet"):
            # Batch-read: filters DTE + downcasts per-batch — never loads full file
            df = _load_parquet_batched(path)
            # Column names already cleaned by the batch loader; just validate
            _check_required_columns(df)
        else:
            # CSV: chunked read with same per-chunk DTE filter
            chunks = []
            for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
                chunk = _clean_column_names(chunk)
                chunk = _resolve_aliases(chunk)
                if "dte" in chunk.columns:
                    chunk["dte"] = pd.to_numeric(chunk["dte"], errors="coerce")
                    chunk = chunk[
                        (chunk["dte"] >= FULL_CHAIN_DTE_MIN) &
                        (chunk["dte"] <= FULL_CHAIN_DTE_MAX)
                    ]
                chunk = _downcast_floats(chunk)
                if not chunk.empty:
                    chunks.append(chunk)
            if not chunks:
                raise ValueError("No rows survived DTE filter in CSV.")
            df = pd.concat(chunks, ignore_index=True)
            _check_required_columns(df)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read '{path}'.\n"
            f"Original error: {e}\n"
            f"Ensure the file exists and is a valid parquet or CSV."
        ) from e

    # ── contract size (optional columns) ─────────────────────────────────────
    # FIX 4: Only parse c_size/p_size if the columns actually exist.
    for col in ("c_size", "p_size"):
        if col in df.columns:
            df[col] = df[col].apply(_parse_contract_size)

    # ── datetime columns ──────────────────────────────────────────────────────
    date_cols = [c for c in df.columns if "date" in c or "readtime" in c]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── numeric columns ───────────────────────────────────────────────────────
    skip = set(date_cols) | {"quote_unixtime", "expire_unix"}
    df = _safe_numeric_cast(df, skip_cols=list(skip))

    # ── DTE (compute if absent) ───────────────────────────────────────────────
    df = _compute_dte_if_missing(df)                       # FIX 3
    df["dte"] = pd.to_numeric(df["dte"], errors="coerce")

    # ── FIX 5: guard against zero/negative strikes before np.log ─────────────
    # Log of zero or a negative number produces -inf / NaN and silently
    # corrupts the moneyness filter downstream.
    bad_strikes = (df["strike"] <= 0) | df["strike"].isna()
    if bad_strikes.any():
        logger.warning(
            f"Dropping {bad_strikes.sum():,} rows with zero/null/negative strike."
        )
        df = df[~bad_strikes]

    # ── FIX 6: guard against zero/negative underlying before np.log ──────────
    bad_ul = (df["underlying_last"] <= 0) | df["underlying_last"].isna()
    if bad_ul.any():
        logger.warning(
            f"Dropping {bad_ul.sum():,} rows with zero/null/negative underlying_last."
        )
        df = df[~bad_ul]

    # ── derived columns ───────────────────────────────────────────────────────
    df["moneyness"]     = np.log(df["underlying_last"] / df["strike"])
    df["abs_moneyness"] = df["moneyness"].abs()

    logger.info(
        f"Raw data loaded: {len(df):,} rows, "
        f"{df['quote_date'].nunique()} dates."
    )
    return df


def build_full_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full options chain for the backtest engine (options_df).

    Filters:
      • Both bid AND ask > 0  (contract is tradeable)
      • DTE in [FULL_CHAIN_DTE_MIN, FULL_CHAIN_DTE_MAX]

    Does NOT filter on IV — a contract can be tradeable even if the IV
    solver returned NaN (deep ITM, wide spread). The backtest can price
    it via bid/ask.

    Keeps strike as float64 so MultiIndex.loc() lookups work correctly.
    """
    # FIX 7: Check for required bid/ask columns before filtering
    bid_ask_cols = ["c_bid", "c_ask", "p_bid", "p_ask"]
    missing_ba = [c for c in bid_ask_cols if c not in df.columns]
    if missing_ba:
        raise KeyError(
            f"build_full_chain: missing bid/ask columns {missing_ba}. "
            f"Check COLUMN_ALIASES or your vendor's column naming."
        )

    mask = (
        (df["c_bid"] > 0) & (df["c_ask"] > 0) &
        (df["p_bid"] > 0) & (df["p_ask"] > 0) &
        (df["dte"] >= FULL_CHAIN_DTE_MIN) &
        (df["dte"] <= FULL_CHAIN_DTE_MAX)
    )
    chain = df[mask].copy()

    if chain.empty:
        logger.warning(
            "build_full_chain produced an EMPTY DataFrame. "
            f"Check that DTE range [{FULL_CHAIN_DTE_MIN}, {FULL_CHAIN_DTE_MAX}] "
            "matches your data and that bid/ask columns are not all zero/NaN."
        )

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

    FIX 8: c_iv / p_iv may not exist in every dataset. If absent, the
    IV filter is skipped with a warning rather than raising a KeyError.
    """
    iv_filter = pd.Series(True, index=df.index)
    if "c_iv" in df.columns and "p_iv" in df.columns:
        iv_filter = df["c_iv"].notna() & df["p_iv"].notna()
    else:
        logger.warning(
            "c_iv / p_iv columns not found — skipping IV filter in signal slice. "
            "Signal features that depend on implied volatility will be NaN."
        )

    mask = (
        (df["dte"] >= SIGNAL_DTE_MIN) &
        (df["dte"] <= SIGNAL_DTE_MAX) &
        (df["abs_moneyness"] < SIGNAL_MONEYNESS_CUTOFF) &
        iv_filter &
        (df["c_bid"] > 0) & (df["c_ask"] > 0) &
        (df["p_bid"] > 0) & (df["p_ask"] > 0)
    )
    signal = df[mask].copy()

    if signal.empty:
        logger.warning(
            "build_signal_data produced an EMPTY DataFrame. "
            f"Check DTE range [{SIGNAL_DTE_MIN}, {SIGNAL_DTE_MAX}], "
            f"moneyness cutoff {SIGNAL_MONEYNESS_CUTOFF}, and IV column presence."
        )

    logger.info(
        f"Signal slice: {len(signal):,} rows "
        f"({signal['quote_date'].nunique()} dates)."
    )
    return signal.reset_index(drop=True)


def _validate(full_chain: pd.DataFrame, signal_data: pd.DataFrame) -> None:
    """Sanity checks that catch common bugs at pipeline time."""

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

    # Check 3: spot-check first 5 signal dates have full_chain coverage
    sample_dates = sorted(signal_dates)[:5]
    for d in sample_dates:
        day_chain = full_chain[full_chain["quote_date"].dt.date == d]
        n = len(day_chain)
        if n == 0:
            logger.warning(f"No full_chain rows for signal date {d}")
        else:
            logger.debug(f"  {d}: {n} contracts in full_chain ✓")

    # Check 4: DTE coverage
    min_dte = full_chain["dte"].min()
    if min_dte > SIGNAL_DTE_MIN:
        logger.warning(
            f"full_chain min DTE is {min_dte}, but signal entries can have "
            f"DTE as low as {SIGNAL_DTE_MIN}. You will get data gaps."
        )

    logger.info("Validation complete.")


def load_full_chain(path: str = "data/processed/full_chain.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["quote_date"]  = pd.to_datetime(df["quote_date"])
    df["expire_date"] = pd.to_datetime(df["expire_date"])
    df["strike"]      = df["strike"].astype(np.float64)
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
    End-to-end pipeline. Returns (full_chain, signal_data) and writes
    both to parquet (preferred) or CSV.

    Usage
    -----
    full_chain, signal_data = run_pipeline("data/raw/spy_2020_2022.csv")
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # FIX 9: Wrap each stage with a try/except so you know exactly where it
    # died rather than getting a generic traceback from deep inside pandas.
    try:
        raw = load_raw(raw_path)
    except Exception as e:
        logger.error(
            f"load_raw() failed.\n"
            f"Run inspect_schema('{raw_path}') to see your file's column names.\n"
            f"Error: {e}"
        )
        raise

    try:
        full_chain = build_full_chain(raw)
    except Exception as e:
        logger.error(f"build_full_chain() failed: {e}")
        raise

    try:
        signal_data = build_signal_data(raw)
    except Exception as e:
        logger.error(f"build_signal_data() failed: {e}")
        raise

    # Prefer parquet (10× faster, preserves dtypes exactly)
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Separate flags from positional path argument
    flags    = {a for a in sys.argv[1:] if a.startswith("--")}
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw_path = pos_args[0] if pos_args else "data/SPY_ALL_YEARS_MASTER.parquet"

    if "--inspect" in flags or "--dry-run" in flags:
        inspect_schema(raw_path)
        sys.exit(0)

    # ── WSL memory guard ──────────────────────────────────────────────────────
    # If you're on WSL and the process still gets killed, add/edit
    # C:\Users\<you>\.wslconfig:
    #   [wsl2]
    #   memory=12GB          # increase to 12 GB (or more)
    #   swap=4GB
    # Then run: wsl --shutdown  (in PowerShell) and restart WSL.
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1024 ** 3
        if avail_gb < 4:
            logger.warning(
                f"Only {avail_gb:.1f} GB RAM available. "
                "The pipeline needs ~4 GB for 13 years of SPY data. "
                "If it crashes, increase WSL memory in ~/.wslconfig (see comment above)."
            )
        else:
            logger.info(f"Available RAM: {avail_gb:.1f} GB  ✓")
    except ImportError:
        pass  # psutil not installed — skip the check

    run_pipeline(raw_path)