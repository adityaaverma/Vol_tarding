import numpy as np
import pandas as pd
import logging
from bs.implied_vol import implied_vol
from bs.pricing import bs_price
logger=logging.getLogger(__name__)
    
def compute_iv_for_chain(df:pd.DataFrame,r:float)->pd.DataFrame:
    '''
    in this function we compute the implied volatilies for the given option chain dataframe. 
    The dataframe is expected to have the following columns:
    1. strike (float)
    2. expiry (datetime)
    3. option_type ('call' or 'put')
    4. bid (float)
    5. ask (float)
    6. lastPrice (float)
    7. underlying_last (float)
    The function will add a new column 'iv' to the dataframe with the computed implied volatilities.
        The 'use' parameter determines which price to use for the implied vol calculation:
        - 'mid': use the mid price (average of bid and ask)
        - 'last': use the last traded price
        - 'best': use the best available price (mid price if available, otherwise last price)
        The function returns the dataframe with the added 'iv' and 'time_to_expiry' columns.

    '''

    copy=df.copy()
    copy=copy.dropna(subset=['K','P','T','option_type','spot'])
    strikes=copy['K'].values
    market_price=copy['P'].values
    time_to_expiries=copy['T'].values
    option_type=copy['option_type'].values
    spot=copy['spot'].values

    iv=implied_vol(market_price,spot,strikes,time_to_expiries,r,option_type)
    copy['iv']=iv
    copy['iv_diff']=copy['iv']-copy['iv_yf']
    copy['moneyness']=np.log(copy['K']/copy['spot'])
    copy['bs_price']=bs_price(spot,strikes,time_to_expiries,r,iv,option_type)
    copy['price_diff']=abs(copy['P']-copy['bs_price'])
    
    print(copy)

    return copy
