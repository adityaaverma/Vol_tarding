import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Optional
from bs.implied_vol import implied_vol
logger=logging.getLogger(__name__)

def _compute_time_to_expiry(expiries:pd.Series,now:Optional[pd.Timestamp]=None)->pd.Series:
    """
    takes in expiries and now as optional arguments and returns time to expiry in years
    """
    if now is None:
        now=pd.Timestamp.utcnow()

    expiries=pd.to_datetime(expiries)
    delta=(expiries-now).dt.total_seconds()
    return np.maximum(delta/(365*24*3600),0)


    
def compute_iv_for_chain(df:pd.DataFrame,r:float,use:str='mid')->pd.DataFrame:
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

    if df.empty:
        return df
    
    df=df.copy()
    df['time_to_expiry']=_compute_time_to_expiry(df['expiry'])
    if use=='mid':
        df['mid']=(df['bid'].fillna(0)+df['ask'].fillna(0))/2
        df['mid']=df['mid'].replace(0,np.nan).fillna(df['lastPrice'])
    elif use=='last':
        df['mid']=df['lastPrice']

    else:
        df['mid']=(df['bid'].fillna(0)+df['ask'].fillna(0))/2
        df['mid']=df['mid'].fillna(df['lastPrice'])

    for z in ['underlying_last','strike','mid','time_to_expiry']:
        df[z]=pd.to_numeric(df[z],errors='coerce')

    market_price=df['mid'].values
    S=df['underlying_last'].values
    K=df['strike'].value
    T=df['time_to_expiry'].values
    option_type=df['option_type'].values

    logger.info('computing implied volatilities for %d options',len(df))
    ivs=implied_vol(market_price,S,K,T,r,option_type)
    df['iv']=ivs

    return df