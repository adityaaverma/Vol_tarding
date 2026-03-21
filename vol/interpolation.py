import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def build_iv_grid(df: pd.DataFrame, n_strikes: int = 80, n_maturities: int = 80):
    """
    Build a smooth IV surface using:
    - moneyness (log(K/S))
    - binning (to remove noise)
    - interpolation

    Returns:
    grid_x (moneyness), grid_t (tenor), grid_iv
    """

    df = df.copy()

    # =========================
    # 🔥 USE MONEINESS (CRITICAL)
    # =========================
    # df['moneyness'] = np.log(df['strike'] / df['underlying_last'])

    # =========================
    # 🔥 BINNING (SMOOTHING STEP)
    # =========================
    df['moneyness_bin'] = pd.cut(df['moneyness'], bins=30)
    df['tenor_bin'] = pd.cut(df['time_to_expiry'], bins=20)

    grouped = (
        df.groupby(['moneyness_bin', 'tenor_bin'])['iv']
        .mean()
        .reset_index()
    )

    # convert bins → numeric
    grouped['moneyness'] = grouped['moneyness_bin'].apply(lambda x: x.mid)
    grouped['tenor'] = grouped['tenor_bin'].apply(lambda x: x.mid)

    grouped = grouped.dropna(subset=['moneyness', 'tenor', 'iv'])

    # =========================
    # 🔥 GRID CREATION
    # =========================
    x = grouped['moneyness'].values
    y = grouped['tenor'].values
    z = grouped['iv'].values

    grid_x, grid_t = np.meshgrid(
        np.linspace(x.min(), x.max(), n_strikes),
        np.linspace(y.min(), y.max(), n_maturities)
    )

    # =========================
    # 🔥 INTERPOLATION
    # =========================
    grid_iv = griddata((x, y), z, (grid_x, grid_t), method='cubic')

    # fallback for NaNs
    nan_mask = np.isnan(grid_iv)
    if np.any(nan_mask):
        grid_iv[nan_mask] = griddata(
            (x, y), z,
            (grid_x[nan_mask], grid_t[nan_mask]),
            method='nearest'
        )

    # =========================
    # 🔥 FINAL CLIP (REMOVE SPIKES)
    # =========================
    grid_iv = np.clip(grid_iv, 0.05, 1.5)

    return grid_x, grid_t, grid_iv
