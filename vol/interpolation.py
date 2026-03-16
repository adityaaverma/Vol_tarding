import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def build_iv_grid(df:pd.DataFrame,n_strikes:int=80,n_maturities:int=80,strike_pad:float=0.1):
    """
    Excepts: 
    This function takes in a dataframe with columns 'strikes', 'time_to_expiry', 
    and 'iv' and builds a grid of implied volatilities using interpolation.
    The grid will have n_strikes number of strike points and n_maturities number 
    of maturity points. The strike range will be padded by strike_pad percentage 
    on both sides to ensure we have a wider range of strikes for interpolation.
    
    Returns:
    The function returns three 2D arrays: grid_strikes, grid_maturities, and 
    grid_iv, which represent the strike prices, time to expiry, and implied 
    volatilities on the grid, respectively.    
    """

    strikes=df['strikes'].values
    maturities=df['time_to_expiry'].values
    ivs=df['iv'].values

    smin,smax=strikes.min(),strikes.max()
    s_pad=max(1.0,strike_pad*(smax-smin))
    grid_strikes=np.linspace(max(1.0,smin-s_pad),smax+s_pad,n_strikes)

    tmin,tmax=maturities.min(),maturities.max()
    grid_maturities=np.linspace(tmin,tmax if tmax>0 else tmin+30/365.0 , n_maturities)

    grid_s,grid_t=np.meshgrid(grid_strikes,grid_maturities)


    pts=np.vstack((strikes,maturities)).T
    grid_iv=griddata(pts,ivs,(grid_s,grid_t),method='cubic')

    nan_masks=np.isnan(grid_iv)
    if(np.any(nan_masks)):
        grid_iv[nan_masks]=griddata(pts,ivs,(grid_s[nan_masks],grid_t[nan_masks]),method='nearest')

    return grid_s,grid_t,grid_iv
