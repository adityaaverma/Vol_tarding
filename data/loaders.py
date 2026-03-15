import yfinance as yf
import pandas as pd
from typing import Literal,Dict
import datetime as dt
import logging
import numpy as np

logger=logging.getLogger(__name__)


def load_option_chain_yahoo(symbol:str)->pd.DataFrame:
    """
    return a DataFrame with cols:
    1. Strike(float),
    2. Expiry(datetime),
    3. OptionType ('call' or 'put')
    4. Bid (float),
    5. Ask (float), 
    6. LastPrice (float),
    7. Underlying_last (float)
    """

    ticker=yf.Ticker(symbol)
    #this fetches list of expiry strings in 'YYYY-MM-DD' format
    expiries=ticker.options
    dfs=[]
    underlying=ticker.history(period='1d')['Close'].iloc[-1]

    for exp in expiries:
        try:
            oc=ticker.option_chain(exp)

        except Exception as e:
            logger.warning("failed to fecth option chain for %s, %s:%s",symbol,exp,e)
            continue

        for which,df in (('call',oc.calls),('put',oc.puts)):
            if df is None or df.empty:
                continue
            temp=df.copy()
            # temp=temp.rename(columns={'lastPrice:'lastPrice','bid':
            temp['expiry']=pd.to_datetime(exp)
            temp['option_type']=which
            temp['underlying_last']=underlying
            dfs.append(temp[['strike','ask','bid','lastPrice','expiry','option_type','underlying_last']])

    if not dfs:
        return pd.DataFrame(columns=['strike','ask','bid','lastPrice','expiry','option_type','underlying_last'])
    
    out=pd.concat(dfs,ignore_index=True)

    out[['strike','ask','bid','lastPrice','underlying_last']]=out[['strike','ask','bid','lastPrice','underlying_last']].appply(pd.to_numeric,errors='coerce')
    return out


