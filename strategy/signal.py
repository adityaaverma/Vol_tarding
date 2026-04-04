import numpy as np
from vol.metrics import iv_rv_spread,z_score
from data.loaders import load_option_chain_yahoo
from vol.realized_vol import forward_realized_vol,realized_vol,log_returns
import yfinance as yf
import pandas as pd

class volSignalEngine:
    def __init__(self,config,data):
        self.window=config.get('window', 5)
        self.symbol=config.get('symbol', 'SPY')
        self.entry_z = config.get('entry_z', 2)
        self.exit_z = config.get('exit_z', 0.5)
        self.data=data
        self.spot_df=None

    def compute_features(self):
        self.data=self.data.sort_values('quote_date')
        self.spot_df = self.data.groupby('quote_date')['underlying_last'].first().reset_index()
        returns = log_returns(self.spot_df['underlying_last'].values)
        self.spot_df['returns'] = np.concatenate([[np.nan], returns])
        self.spot_df['rv']=realized_vol(self.spot_df['underlying_last'].values,self.window)
        self.spot_df['fwd_rv']=forward_realized_vol(self.spot_df['underlying_last'].values,self.window)
        self.data=self.data.merge(self.spot_df[['quote_date','returns','rv','fwd_rv']],on='quote_date',how='left')
        
        pass
    
# V = volSignalEngine(config={'symbol': 'SPY', 'window': 20})

data=pd.read_csv(r'data\SPY_Optionsdata(2019-22).csv')
V = volSignalEngine(config={'symbol': 'SPY', 'window': 2},data=data)
V.compute_features()