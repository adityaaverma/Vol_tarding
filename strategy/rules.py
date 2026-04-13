import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class RuleConfig:
    execution_lag: int = 1          # daily EOD signal -> trade next day
    min_liquidity: float = 0.5      # block trades below this score
    allow_flip: bool = True         # allow short -> long directly if signal changes
    max_holding_days: int | None = None
    cooldown_days: int = 0

class VolTradeRules:
    def __init__(self,config:RuleConfig):
        self.config=config

    def apply(self,df:pd.DataFrame)->pd.DataFrame:
        out=df.copy()

        required_cols=['quote_date','out','signal','signal_side','signal_change','out']
        missing=[col for col in required_cols if col not in out.columns]

        if missing:
            raise ValueError(f"missing required columns: {missing}")
        
        out=out.sort_values('quote_date').reset_index(drop=True)

        raw_position=out['signal'].fillna(0).astype(int)

        if "liquidity_score" in out.columns:
            raw_position=np.where(out['liquidity_score'].to_numpy()>=self.config.min_liquidity,raw_position,0)

            raw_position=pd.Series(raw_position,index=out.index)

        
        executed_position=raw_position.shift(self.config.execution_lag).fillna(0).astype(int)

        if not self.config.allow_flip:
            prev_raw=raw_position.shift(1).fillna(0).astype(int)
            flip_mask=(prev_raw==1) & (raw_position==-1) | (prev_raw==-1) & (raw_position==1)
            raw_position=raw_position.where(~flip_mask,0)
            executed_position=raw_position.shift(self.config.execution_lag).fillna(0).astype(int)

        out['raw_position']=raw_position
        out['position']=executed_position

        out['position_prev']=out['position'].shift(1).fillna(0).astype(int)
        out['entry_flag']=((out['position_prev']==0) & (out['position']!=0)).astype(int)
        out['exit_flag']=((out['position_prev']!=0) & out['position']==0).astype(int)
        out['trade_action']=(out['position']-out['position_prev']).astype(int)

        out['active']=(out['position']!=0).astype(int)

        return out

def run_rules(df: pd.DataFrame, config: RuleConfig | None = None) -> pd.DataFrame:
    rules = VolTradeRules(config or RuleConfig())
    return rules.apply(df)
        





