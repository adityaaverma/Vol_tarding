import pandas as pd
from strategy.sizing import VolSizer
from execution.delta_hedge import DeltaHedgeEngine
from strategy.position import PositionManager




class VolBacktest:
    def __init__(self, data:pd.DataFrame,sizer:VolSizer,hedger:DeltaHedgeEngine):
        self.data=data
        self.sizer=sizer
        self.hedger=hedger
        self.portfolio_value=1_000_000
        self.current_pos=None
        self.shares=0.0

    def run(self):
        results=[]

        for i,row in self.data.iterrows():
            if self.current_pos and row['exit_flag']==1:
                self.current_pos=None
                self.shares=0.0

            if not self.current_pos and row['entry_flag']==1:
                qty=self.sizer.calculate_quantity(row)
                self.current_pos=PositionManager("SPY").create_straddle(row,qty)

            if self.current_pos:
                opt_delta=row['c_delta']+row['p_delta']
                trade_shares=self.hedger.get_hedge_action(self.shares,opt_delta,self.current_pos.quantiy)

            results.append({'date':row['quote_date'],'pnl':0})

        return pd.DataFrame(results)
    

    