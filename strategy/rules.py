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
        





