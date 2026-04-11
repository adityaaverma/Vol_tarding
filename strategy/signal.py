import numpy as np
from vol.metrics import z_score, compute_percentile
from vol.realized_vol import forward_realized_vol,realized_vol,log_returns
import pandas as pd

class volSignalEngine:
    def __init__(self,config):
        self.window=config.get('window', 5)
        self.symbol=config.get('symbol', 'SPY')
        self.entry_z = config.get('entry_z', 2)
        self.exit_z = config.get('exit_z', 0.5)
        self.dailyData=pd.DataFrame()
        self.spot_df=None
        self.config=config

    def compute_features(self,data):

        # 1. Sabse pehle moneyness ke basis par sort karein

        term_df=self._compute_term_structure_score(data)
        skew_df=self._compute_skew(data)
        data_sorted = data.sort_values(['quote_date', 'abs_moneyness'],kind='mergesort')

        # 2. Har 'quote_date' ki pehli row (min moneyness) pick karein
        # groupby().head(1) vectorized hai aur loop se fast hai
        daily_data = data_sorted.groupby('quote_date').head(1).copy()
        daily_data=daily_data.merge(term_df,on='quote_date',how='left')
        daily_data=daily_data.merge(skew_df,on='quote_date',how='left')
        

        # 3. IV Calculation (Vectorized logic)
        # np.select use karke hum if-else ko pure column par ek saath apply kar sakte hain
        conditions = [
            (daily_data['c_iv'].notna()) & (daily_data['p_iv'].notna()),
            (daily_data['c_iv'].notna()),
            (daily_data['p_iv'].notna())
        ]

        choices = [
            (daily_data['c_iv'] + daily_data['p_iv']) / 2,
            daily_data['c_iv'],
            daily_data['p_iv']
        ]
        # choicesskew=[daily_data['p_iv']-daily_data['c_iv'],np.nan,np.nan]
        choices_call_put_iv_gap=[daily_data['c_iv']-daily_data['p_iv'],np.nan,np.nan]

        daily_data['iv'] = np.select(conditions, choices, default=np.nan)
        # daily_data['skew']=np.select(conditions, choicesskew, default=np.nan)
        daily_data['call_put_iv_gap']=np.select(conditions, choices_call_put_iv_gap, default=np.nan)


        # 4. Final result ko self.dailyData mein assign ya append karein
        self.dailyData = pd.concat([self.dailyData, daily_data], ignore_index=True)

        
        returns=log_returns(self.dailyData['underlying_last'].values)
        self.dailyData['returns']=np.concatenate([[np.nan],returns])
        self.dailyData['rv']=realized_vol(self.dailyData['underlying_last'].values,self.window)
        self.dailyData['fwd_rv']=forward_realized_vol(self.dailyData['underlying_last'].values,self.window)
        self.dailyData['spread']=self.dailyData['iv']-self.dailyData['rv']
        self.dailyData['fwd_spread']=self.dailyData['iv']-self.dailyData['fwd_rv']
        self.dailyData['spread_z']=z_score(self.dailyData['spread'].values,self.window)
        self.dailyData['spread_percentile']=compute_percentile(self.dailyData['spread'].values,self.window)
        self.dailyData['iv_change_1d']=self.dailyData['iv'].astype(float).diff(1)
        self.dailyData['iv_change_5d']=self.dailyData['iv'].astype(float).diff(5)
        self.dailyData['rv_change_5d']=self.dailyData['rv'].astype(float).diff(5)
        self.dailyData['liquidity_score']=self._compute_liquidity_score(self.dailyData)
        self.dailyData['term_slope']=z_score(self.dailyData['term_slope_raw'].values,self.window)
        self.dailyData['regime_score']=self._compute_regime_score(self.dailyData['rv'])
        self.dailyData['skew_z']=z_score(self.dailyData['skew'].values,self.window)
       
        self.dailyData.dropna(subset=['spread_z','spread_percentile'],inplace=True)
        # print(self.dailyData.columns)
        spread_pct=self.dailyData['spread_percentile'].astype(float).to_numpy()
        spread_z=self.dailyData['spread_z'].astype(float).to_numpy()
        skew_z=self.dailyData['skew_z'].astype(float).to_numpy()
        term_slope=self.dailyData['term_slope'].astype(float).to_numpy()
        regime=self.dailyData['regime_score'].astype(float).to_numpy()
        liq=self.dailyData['liquidity_score'].astype(float).to_numpy()

        composite=(self.config['w_spread_z']*spread_z + 
                   self.config['w_spread_pctile']*(spread_pct-0.5) + 
                   self.config['w_skew']*np.nan_to_num(skew_z,nan=0.0) +
                   self.config['w_term_slope']*np.nan_to_num(term_slope,nan=0.0) +
                   self.config['w_regime']*np.nan_to_num(regime,nan=0.0) +
                   self.config['w_liquidity']*np.nan_to_num(liq-0.5,nan=0.0))
        
        self.dailyData['out']=composite
        print(self.dailyData.loc[100:110, ['quote_date',
                                        'moneyness', 
                                        'c_iv', 
                                        'p_iv', 
                                        'iv',
                                        'underlying_last',
                                    'strike',
                                    'returns',
                                        'rv',
                                        'fwd_rv',
                                        'spread',
                                        'fwd_spread',
                                        # 'z_score',
                                        # 'z_percentile',
                                        # 'iv_change_1d',
                                        #      'iv_change_5d',
                                            #    'rv_change_5d',
                                            #    'liquidity_score',
                                                # 'term_slope',
                                                # 'regime_score',
                                                # 'iv_temp'
                                                'skew',
                                                'out'
                                                ]])
        print(self.dailyData['out'].describe())
        
        

    def _compute_liquidity_score(self,daily:pd.DataFrame)->pd.Series:

        parts=[]

        if 'c_volume' in daily.columns and 'p_volume' in daily.columns:
            volume=pd.to_numeric(daily['c_volume'],errors='coerce').fillna(0)
            +pd.to_numeric(daily['p_volume'],errors='coerce').fillna(0)

            volume=np.log1p(volume)
            volume_z_score=z_score(volume.values,self.window)
            parts.append(volume_z_score)


        if all(col in daily.columns for col in ["c_bid", "c_ask", "p_bid", "p_ask"]):
            c_spread=pd.to_numeric(daily['c_ask'],errors='coerce')-pd.to_numeric(daily['c_bid'],errors='coerce')
            p_spread=pd.to_numeric(daily['p_ask'],errors='coerce')-pd.to_numeric(daily['p_bid'],errors='coerce')

            spread=(c_spread+p_spread)/2    
            inv_spread = np.where(np.isfinite(spread) & (spread > 0), 1.0 / spread, np.nan)
            inv_z_score=z_score(inv_spread,self.window)
            parts.append(inv_z_score)

        
        arr=np.vstack([np.asarray(p,dtype=float) for p in parts])
        liquidity=np.nanmean(arr,axis=0)
        liquidity=np.clip(liquidity, -3, 3)
        liquidity=1.0/(1.0+np.exp(-liquidity))
        return pd.Series(liquidity , index=daily.index)
    
    def _compute_term_structure_score(self,data:pd.DataFrame)->pd.DataFrame:
        term_list=[]
        copy=data.copy()
        for date,group in copy.groupby('quote_date'):
            group['dte']=pd.to_numeric(group['dte'],errors='coerce')
            group['iv_temp']=np.where(group['c_iv'].notna() & group['p_iv'].notna(),
                                      (group['c_iv']+group['p_iv'])/2,
                                      np.where(group['c_iv'].notna(),group['c_iv'],group['p_iv']))
            
            near_iv=group[group['dte'].between(20,30)]['iv_temp'].mean()
            far_iv=group[group['dte'].between(45,90)]['iv_temp'].mean()
            if np.isfinite(near_iv) and np.isfinite(far_iv):
                slope=far_iv-near_iv
            else:
                slope=np.nan
            term_list.append((date,slope))

        term_df=pd.DataFrame(term_list,columns=['quote_date','term_slope_raw'])
        return term_df

        
    def _compute_regime_score(self,rv:pd.Series)->pd.Series:
        rv=pd.to_numeric(rv,errors='coerce').astype(float).to_numpy()
        rv_z=z_score(rv,self.window)
        regime = -np.tanh(np.nan_to_num(rv_z,nan=0.0)/2.0)
        return pd.Series(regime,index=self.dailyData.index)
    
    def _compute_skew(self,data:pd.DataFrame)->pd.Series:

        skew_list=[]
        copy=data.copy()
        for date,group in copy.groupby('quote_date'):
            put_iv = group[(group['moneyness'] < 0) & (group['abs_moneyness'] < 0.05)].sort_values('abs_moneyness')['p_iv'].iloc[0] if not group[group["moneyness"] < 1].empty else np.nan
            call_iv = group[(group['moneyness'] > 0) & (group['abs_moneyness'] < 0.05)].sort_values('abs_moneyness')['c_iv'].iloc[0] if not group[group["moneyness"] > 1].empty else np.nan
            skew=put_iv-call_iv if np.isfinite(put_iv) and np.isfinite(call_iv) else np.nan
            skew_list.append((date,skew))
        
        skew_df=pd.DataFrame(skew_list,columns=['quote_date','skew'])
        return skew_df


        

# data=pd.read_csv(r'data\SPY_Optionsdata(2019-22).csv')
data=pd.read_csv(r'data\dte90.csv')
print(len(data))
data['moneyness']=np.log(data['underlying_last']/data['strike'])
data['abs_moneyness']=data['moneyness'].abs()
V = volSignalEngine(config={'symbol': 'SPY', 
                            'window': 30, 
                            'w_spread_z' : 0.45,
                            'w_spread_pctile' : 0.15,
                            'w_skew' : 0.10,
                            'w_term_slope' : 0.10,
                            'w_regime' : 0.10,
                            'w_liquidity' : 0.10})

V.compute_features(data)
# print(data.loc[:,['moneyness']])