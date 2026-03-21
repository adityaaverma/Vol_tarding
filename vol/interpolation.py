import numpy as np
import pandas as pd
from scipy.interpolate import griddata,UnivariateSpline

from scipy.interpolate import interp1d

from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
import pandas as pd

def build_iv_grid(df, n_strikes=80, n_maturities=80):

    df = df.copy()
    smiles = {}

    # =========================
    # 🔥 STEP 1: Build smiles
    # =========================
    for expiry, group in df.groupby('expiry'):

        group = group.sort_values('moneyness')

        # ✅ remove duplicates
        group = group.drop_duplicates(subset='moneyness')

        k = group['moneyness'].values
        iv = group['iv'].values
        T = group['time_to_expiry'].median()

        if len(k) < 3:
            continue

        try:
            # 🔥 stable smoothing
            spline = UnivariateSpline(k, iv, s=0.05)

            k_grid = np.linspace(k.min(), k.max(), n_strikes)
            iv_smooth = spline(k_grid)

            # ✅ clip to realistic range
            iv_smooth = np.clip(iv_smooth, 0.05, 1.0)

        except Exception as e:
            print(f"Spline failed for expiry {expiry}: {e}")

            # 🔥 fallback → raw data
            k_grid = k
            iv_smooth = iv

        smiles[T] = (k_grid, iv_smooth)

    if len(smiles) < 2:
        raise ValueError("Not enough valid expiries")

    # =========================
    # 🔥 STEP 2: ALIGN STRIKES (NO EXTRAPOLATION)
    # =========================
    common_k = np.linspace(
        min(v[0].min() for v in smiles.values()),
        max(v[0].max() for v in smiles.values()),
        n_strikes
    )

    tenors = sorted(smiles.keys())
    iv_matrix = []

    for T in tenors:
        k_grid, iv_vals = smiles[T]

        f = interp1d(
            k_grid,
            iv_vals,
            bounds_error=False,
            fill_value=np.nan   # ❌ NO extrapolation
        )

        iv_matrix.append(f(common_k))

    iv_matrix = np.array(iv_matrix)

    # =========================
    # 🔥 STEP 3: INTERPOLATE ACROSS TIME (NO EXTRAPOLATION)
    # =========================
    T_grid = np.linspace(min(tenors), max(tenors), n_maturities)

    final_surface = []

    for i in range(len(common_k)):
        f = interp1d(
            tenors,
            iv_matrix[:, i],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan   # ❌ NO extrapolation
        )
        final_surface.append(f(T_grid))

    final_surface = np.array(final_surface).T

    # =========================
    # 🔥 STEP 4: DEBUG (NO FAKE FILL)
    # =========================
    print(
        "Surface stats:",
        np.nanmin(final_surface),
        np.nanmax(final_surface),
        np.nanstd(final_surface),
        "NaNs:", np.isnan(final_surface).sum()
    )

    # Optional: mask NaNs for cleaner plotting
    final_surface = np.ma.masked_invalid(final_surface)

    grid_x, grid_t = np.meshgrid(common_k, T_grid)

    return grid_x, grid_t, final_surface
