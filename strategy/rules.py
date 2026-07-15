import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

logger=logging.getLogger(__name__)

REQUIRED_COLS=['quote_date','out','signal','signal_side','signal_change']
@dataclass
class RuleConfig:
    execution_lag: int = 1          # daily EOD signal -> trade next day
    min_liquidity: float = 0.5      # block trades below this score
    allow_flip: bool = True         # allow short -> long directly if signal changes
    min_holding_days: int = 5
    max_holding_days: int | None = 21
    cooldown_days: int = 0

class VolTradeRules:
    def __init__(self,config:RuleConfig):
        self.config=config

    def validate(self,df:pd.DataFrame)->None:
        missing=[c for c in REQUIRED_COLS if c  not in df.columns]
        if missing:
            raise ValueError(f"backtest cannot proceed: missing critical columns: {missing}")
        

    def apply(self,df:pd.DataFrame)->pd.DataFrame:
        out=df.copy()
        self.validate(out)
        out=out.sort_values('quote_date').reset_index(drop=True)

        # 1. Vectorized liquidity Gate
        # Instituional trade: if market is too thin we simply do not trade
        raw_position=out['signal'].fillna(0).astype(int)
        low_liq_mask=pd.Series(False,index=out.index)
        if "liquidity_score" in out.columns:
            low_liq_mask=out['liquidity_score']<self.config.min_liquidity
            if low_liq_mask.any():
                logger.info(f"Liquidity gate supressed trades on {low_liq_mask.sum()} bars.")
        
        #vectorized flip gaurd
        if not self.config.allow_flip:
            prev_raw=raw_position.shift(1).fillna(0).astype(int)
            flip_mask=((prev_raw==1) & (raw_position==-1)) | ((prev_raw==-1) & (raw_position==1))
            raw_position[flip_mask]=0   
        
        executed_position=raw_position.shift(self.config.execution_lag).fillna(0).astype(int)
        low_liq_shifted=low_liq_mask.shift(self.config.execution_lag).fillna(False)
        final_position=self._apply_stateful_rules(out,executed_position,low_liq_shifted)

        out['position']=final_position
        out['position_prev']=out['position'].shift(1).fillna(0).astype(int)
        out['entry_flag']=((out['position_prev']==0) & (out['position']!=0)).astype(int)
        out['exit_flag']=((out['position_prev']!=0) & (out['position']==0)).astype(int)
        out['trade_action']=(out['position']-out['position_prev']).astype(int)

        out['active']=(out['position']!=0).astype(int)

        return out
    
    def _apply_stateful_rules(self,df:pd.DataFrame, executed:pd.Series,low_liq_mask:pd.Series)->pd.Series:
        """
        enforces path dependent logic:
        cooldown: no new entries after N days of exit
        max Holding: Foorces close if we've held same contract for too long
        """

        dates=pd.to_datetime(df['quote_date']).to_numpy()
        pos_arr=executed.to_numpy().copy()
        liq_blocked=low_liq_mask.to_numpy()
        
        current_pos=0
        entry_date=None
        cooldown_end_idx=-1

        for i in range(len(pos_arr)):

            desired=pos_arr[i]
            #Cooldown check
            if i<=cooldown_end_idx and desired!=0 and current_pos==0:
                desired=0
            
            #liquidity - gate
            if current_pos==0 and desired!=0 and liq_blocked[i]:
                desired=0

            #Max holding check
            if current_pos!=0 and entry_date is not None:
                days_held=(dates[i]-entry_date).astype('timedelta64[D]').astype(int)
                if self.config.max_holding_days and self.config.max_holding_days<=days_held:
                    logger.debug(f"max holding dates resched at index {i}. Forcing exit")
                    desired=0
                if self.config.min_holding_days and days_held < self.config.min_holding_days:
                    desired = current_pos # suppress early exit

            # State transition Machines
            # Entry
            if current_pos==0 and desired!=0:
                current_pos=desired
                entry_date=dates[i]

            # Exit
            elif current_pos!=0 and desired==0:
                current_pos=0
                entry_date=None
                cooldown_end_idx=i+self.config.cooldown_days
            
            # flip allowed
            elif current_pos!=0 and desired!=current_pos:
                current_pos=desired
                entry_date=dates[i]

            pos_arr[i]=desired
        
        return pd.Series(pos_arr,index=executed.index,dtype=int)

            

def run_rules(df: pd.DataFrame, config: RuleConfig | None = None) -> pd.DataFrame:
    rules = VolTradeRules(config or RuleConfig())
    return rules.apply(df)
        





