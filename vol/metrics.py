import numpy as np

def iv_rv_spread(iv:np.ndarray,rv:np.ndarray)->np.ndarray:
    '''
    Compute the spread between implied volatility and realized volatility.
    The spread is calculated as the difference between the implied volatility and the realized volatility.
    '''
    if len(iv) != len(rv):
        raise ValueError("Input arrays must have the same length.")
    
    return iv - rv


def z_score(series:np.ndarray,window:int)->np.ndarray:
    '''
    Compute the z-score of a series using a rolling window approach.
    '''
    kernel=np.ones(window)/window
    Ex=np.convolve(series,kernel,'valid')
    Ex2=np.convolve(series**2,kernel,'valid')
    var=Ex2-Ex**2
    std=np.sqrt(var)
    std[std==0]=np.nan
    z=(series[window-1:]-Ex)/std
    z_full=np.full(len(series),np.nan)
    z_full[window-1:]=z
    return z_full



a=np.array([1, 2, 3, 4, 5,6,7,8,9,10])
z=z_score(a,3)
print(z)