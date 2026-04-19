import numpy as np
from vol.metrics import z_score
from vol.realized_vol import forward_realized_vol,realized_vol,log_returns
import pandas as pd

class volSignalEngine:
    def __init__(self,config):
        self.window=config.get('window', 5)
        self.symbol=config.get('symbol', 'SPY')
        self.entry_z = config.get('entry_z', 2)
        self.exit_z = config.get('exit_z', 0.5)
        self.signal_mode = config.get('signal_mode', 'short_rich_vol')
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
        self.dailyData['rv']=self._safe_realized_vol(self.dailyData['underlying_last'].values)
        self.dailyData['fwd_rv']=self._safe_forward_realized_vol(self.dailyData['underlying_last'].values)
        self.dailyData['spread']=self.dailyData['iv']-self.dailyData['rv']
        self.dailyData['fwd_spread']=self.dailyData['iv']-self.dailyData['fwd_rv']
        self.dailyData['spread_z']=self._safe_z_score(self.dailyData['spread'].values)
        self.dailyData['spread_percentile']=self._safe_percentile(self.dailyData['spread'].values)
        self.dailyData['iv_change_1d']=self.dailyData['iv'].astype(float).diff(1)
        self.dailyData['iv_change_5d']=self.dailyData['iv'].astype(float).diff(5)
        self.dailyData['rv_change_5d']=self.dailyData['rv'].astype(float).diff(5)
        self.dailyData['liquidity_score']=self._compute_liquidity_score(self.dailyData)
        self.dailyData['term_slope']=self._safe_z_score(self.dailyData['term_slope_raw'].values)
        self.dailyData['regime_score']=self._compute_regime_score(self.dailyData['rv'])
        self.dailyData['skew_z']=self._safe_z_score(self.dailyData['skew'].values)
       
        self.dailyData.dropna(subset=['spread_z','spread_percentile'],inplace=True)
        # print(self.dailyData.columns)
        spread_pct=self.dailyData['spread_percentile'].astype(float).to_numpy()
        spread_z=self.dailyData['spread_z'].astype(float).to_numpy()
        skew_z=self.dailyData['skew_z'].astype(float).to_numpy()
        term_slope=self.dailyData['term_slope'].astype(float).to_numpy()
        regime=self.dailyData['regime_score'].astype(float).to_numpy()
        liq=self.dailyData['liquidity_score'].astype(float).to_numpy()

        contribution_df = self._compute_contributions(
            spread_z=spread_z,
            spread_pct=spread_pct,
            skew_z=skew_z,
            term_slope=term_slope,
            regime=regime,
            liquidity=liq,
        )
        for column in contribution_df.columns:
            self.dailyData[column] = contribution_df[column].to_numpy()

        composite=self.dailyData['spread_z_contrib'].to_numpy() + \
                  self.dailyData['spread_pctile_contrib'].to_numpy() + \
                  self.dailyData['skew_contrib'].to_numpy() + \
                  self.dailyData['term_slope_contrib'].to_numpy() + \
                  self.dailyData['regime_contrib'].to_numpy() + \
                  self.dailyData['liquidity_contrib'].to_numpy()

        self.dailyData['out']=composite
        signal_df = self._generate_signal(self.dailyData['out'].to_numpy())
        for column in signal_df.columns:
            self.dailyData[column] = signal_df[column].to_numpy()
        return self.dailyData.copy()
        
        

    def _compute_liquidity_score(self,daily:pd.DataFrame)->pd.Series:

        parts=[]

        if 'c_volume' in daily.columns and 'p_volume' in daily.columns:
            volume= pd.to_numeric(daily['c_volume'],errors='coerce').fillna(0) + pd.to_numeric(daily['p_volume'],errors='coerce').fillna(0)
            volume=np.log1p(volume)
            volume_z_score=self._safe_z_score(volume.values)
            parts.append(volume_z_score)


        if all(col in daily.columns for col in ["c_bid", "c_ask", "p_bid", "p_ask"]):
            c_spread=pd.to_numeric(daily['c_ask'],errors='coerce')-pd.to_numeric(daily['c_bid'],errors='coerce')
            p_spread=pd.to_numeric(daily['p_ask'],errors='coerce')-pd.to_numeric(daily['p_bid'],errors='coerce')

            spread=(c_spread+p_spread)/2    
            inv_spread = np.where(np.isfinite(spread) & (spread > 0), 1.0 / spread, np.nan)
            inv_z_score=self._safe_z_score(inv_spread)
            parts.append(inv_z_score)

        if not parts:
            return pd.Series(np.full(len(daily), 0.5), index=daily.index, dtype=float)

        arr=np.vstack([np.asarray(p,dtype=float) for p in parts])
        valid_count=np.sum(np.isfinite(arr),axis=0)
        liquidity=np.divide(
            np.nansum(arr,axis=0),
            valid_count,
            out=np.zeros(arr.shape[1],dtype=float),
            where=valid_count > 0,
        )
        liquidity=np.clip(liquidity, -3, 3)
        liquidity=1.0/(1.0+np.exp(-liquidity))
        liquidity=np.where(valid_count > 0, liquidity, 0.5)
        return pd.Series(liquidity , index=daily.index)
    
    def _compute_term_structure_score(self,data:pd.DataFrame)->pd.DataFrame:
        term_list=[]
        copy=data.copy()
        for date,group in copy.groupby('quote_date'):
            group['dte']=pd.to_numeric(group['dte'],errors='coerce')
            group['iv_temp']=np.where(group['c_iv'].notna() & group['p_iv'].notna(),
                                      (group['c_iv']+group['p_iv'])/2,
                                      np.where(group['c_iv'].notna(),group['c_iv'],group['p_iv']))
            
            near_slice = group[group['dte'].between(20,30)]['iv_temp'].dropna()
            far_slice = group[group['dte'].between(45,90)]['iv_temp'].dropna()

            near_iv = near_slice.mean() if not near_slice.empty else np.nan
            far_iv = far_slice.mean() if not far_slice.empty else np.nan
            if np.isfinite(near_iv) and np.isfinite(far_iv):
                slope=far_iv-near_iv
            else:
                slope=np.nan
            term_list.append((date,slope))

        term_df=pd.DataFrame(term_list,columns=['quote_date','term_slope_raw'])
        return term_df

        
    def _compute_regime_score(self,rv:pd.Series)->pd.Series:
        rv=pd.to_numeric(rv,errors='coerce').astype(float).to_numpy()
        rv_z=self._safe_z_score(rv)
        regime = -np.tanh(np.nan_to_num(rv_z,nan=0.0)/2.0)
        return pd.Series(regime,index=self.dailyData.index)
    
    def _compute_skew(self,data:pd.DataFrame)->pd.Series:

        skew_list=[]
        copy=data.copy()
        for date,group in copy.groupby('quote_date'):
            put_slice=group[(group['moneyness'] < 0) & (group['abs_moneyness'] < 0.05)].sort_values('abs_moneyness')
            call_slice=group[(group['moneyness'] > 0) & (group['abs_moneyness'] < 0.05)].sort_values('abs_moneyness')

            put_iv = put_slice['p_iv'].iloc[0] if not put_slice.empty else np.nan
            call_iv = call_slice['c_iv'].iloc[0] if not call_slice.empty else np.nan
            skew=put_iv-call_iv if np.isfinite(put_iv) and np.isfinite(call_iv) else np.nan
            skew_list.append((date,skew))
        
        skew_df=pd.DataFrame(skew_list,columns=['quote_date','skew'])
        return skew_df

    def _compute_contributions(
        self,
        spread_z: np.ndarray,
        spread_pct: np.ndarray,
        skew_z: np.ndarray,
        term_slope: np.ndarray,
        regime: np.ndarray,
        liquidity: np.ndarray,
    ) -> pd.DataFrame:
        spread_z_contrib = self.config['w_spread_z'] * spread_z
        spread_pctile_contrib = self.config['w_spread_pctile'] * (spread_pct - 0.5)
        skew_contrib = self.config['w_skew'] * np.nan_to_num(skew_z, nan=0.0)
        term_slope_contrib = self.config['w_term_slope'] * np.nan_to_num(term_slope, nan=0.0)
        regime_contrib = self.config['w_regime'] * np.nan_to_num(regime, nan=0.0)
        liquidity_contrib = self.config['w_liquidity'] * np.nan_to_num(liquidity - 0.5, nan=0.0)

        return pd.DataFrame({
            'spread_z_contrib': spread_z_contrib,
            'spread_pctile_contrib': spread_pctile_contrib,
            'skew_contrib': skew_contrib,
            'term_slope_contrib': term_slope_contrib,
            'regime_contrib': regime_contrib,
            'liquidity_contrib': liquidity_contrib,
        })

    def _generate_signal(self, out: np.ndarray) -> pd.DataFrame:
        if len(out) == 0:
            return pd.DataFrame({
                'signal': pd.Series(dtype=int),
                'signal_side': pd.Series(dtype=object),
                'signal_change': pd.Series(dtype=int),
            })

        if self.signal_mode == 'long_rich_vol':
            long_entry = out >= self.entry_z
            short_entry = out <= -self.entry_z
            long_exit = out <= self.exit_z
            short_exit = out >= -self.exit_z
        else:
            long_entry = out <= -self.entry_z
            short_entry = out >= self.entry_z
            long_exit = out >= -self.exit_z
            short_exit = out <= self.exit_z

        signal = np.zeros(len(out), dtype=int)
        position = 0

        for i in range(len(out)):
            value = out[i]
            if not np.isfinite(value):
                signal[i] = position
                continue

            if position == 0:
                if long_entry[i]:
                    position = 1
                elif short_entry[i]:
                    position = -1
            elif position == 1:
                if long_exit[i]:
                    position = 0
            elif position == -1:
                if short_exit[i]:
                    position = 0

            signal[i] = position

        signal_side = np.where(
            signal > 0,
            'long_vol',
            np.where(signal < 0, 'short_vol', 'flat'),
        )
        signal_change = np.concatenate([[0], np.diff(signal)])

        return pd.DataFrame({
            'signal': signal,
            'signal_side': signal_side,
            'signal_change': signal_change,
        })

    def _safe_z_score(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size < self.window:
            return np.full(arr.shape[0], np.nan, dtype=float)
        return z_score(arr, self.window)

    def _safe_realized_vol(self, prices: np.ndarray) -> np.ndarray:
        arr = np.asarray(prices, dtype=float)
        if arr.size <= self.window:
            return np.full(arr.shape[0], np.nan, dtype=float)
        return realized_vol(arr, self.window)

    def _safe_forward_realized_vol(self, prices: np.ndarray) -> np.ndarray:
        arr = np.asarray(prices, dtype=float)
        if arr.size <= self.window:
            return np.full(arr.shape[0], np.nan, dtype=float)
        return forward_realized_vol(arr, self.window)

    def _safe_percentile(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        percentile = np.full(arr.shape[0], np.nan, dtype=float)

        for i in range(self.window - 1, len(arr)):
            window_data = arr[i - self.window + 1:i + 1]
            valid = np.isfinite(window_data)
            if not np.any(valid):
                continue
            last = window_data[-1]
            if not np.isfinite(last):
                continue
            percentile[i] = np.mean(window_data[valid] <= last)

        return percentile


def run_signal_pipeline(
    data: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    prepared = data.copy()
    prepared['moneyness']=np.log(prepared['underlying_last']/prepared['strike'])
    prepared['abs_moneyness']=prepared['moneyness'].abs()

    engine = volSignalEngine(config=config or {
        'symbol': 'SPY',
        'window': 30,
        'entry_z': 1.0,
        'exit_z': 0.25,
        'signal_mode': 'short_rich_vol',
        'w_spread_z': 0.45,
        'w_spread_pctile': 0.15,
        'w_skew': 0.10,
        'w_term_slope': 0.10,
        'w_regime': 0.10,
        'w_liquidity': 0.10,
    })
    return engine.compute_features(prepared)


if __name__ == '__main__':
    data=pd.read_csv(r'data\dte90.csv')
    print(len(data))
    result = run_signal_pipeline(data)
    print(result.loc[100:150, ['quote_date',
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
                               'skew',
                               'out']])
    print(result['out'].describe())
    print(result[['spread_z','term_slope','skew','out']].corr())
    print(result[['quote_date', 'out', 'signal', 'signal_side', 'signal_change']].tail(10))
    print(result.columns)


