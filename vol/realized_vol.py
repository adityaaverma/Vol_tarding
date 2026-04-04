import numpy as np 


def log_returns(prices:np.ndarray)->np.ndarray:
    '''
    Compute log returns from a series of prices.
    '''
    return np.diff(np.log(prices))

def realized_vol(prices:np.ndarray,window:int)->np.ndarray:
    '''
    Compute the realized volatility from a series of prices using a rolling window approach.
    The realized volatility is calculated as the square root of the sum of squared log returns over the
    specified window, annualized by multiplying by the square root of 252 (the number of trading days in a year).
    '''
    if window<=0:
        raise ValueError("Window size must be positive.")
    returns=log_returns(prices)**2
    
    kernel=np.ones(window)/window
    con=np.convolve(returns,kernel,'valid')
    rv=np.sqrt(con*252)
    rv_full = np.full(len(prices), np.nan)
    rv_full[window:] = rv
    return rv_full


def forward_realized_vol(prices: np.ndarray, window: int) -> np.ndarray:
    '''
    Compute the forward-looking realized volatility from a series of prices using a rolling window approach.
    The forward-looking realized volatility is calculated as the square root of the sum of squared log returns over the
    specified window, annualized by multiplying by the square root of 252 (the number of trading days in a year). 
    The function shifts the realized volatility backward to align with the starting point of the window.
    '''
    returns = log_returns(prices)**2
    
    kernel = np.ones(window) / window
    rolling_mean = np.convolve(returns, kernel, 'valid')
    
    rv = np.sqrt(252 * rolling_mean)
    
    rv_full = np.full(len(prices), np.nan)
    
    #  (shift backward)
    rv_full[:-window] = rv
    
    return rv_full


# prices=np.exp(np.cumsum(np.random.normal(0,0.01,1000)))

# print(realized_vol(prices,window=30))