import pandas as pd
from strategy.sizing import VolSizer
from execution.delta_hedge import DeltaHedgeEngine
from strategy.position import PositionManager,StraddlePosition
from execution.transactions_costs import CostConfig,TransactionalCostModel
from typing import Optional
from dataclasses import dataclass,field
import logging

logger=logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    initial_capital:float=0.0
    costConfig:CostConfig =field(default_factory=CostConfig)
    multiplier:float=100
    ticker:str="SPY"

    #Risk Management Gaurds
    stop_loss_pct:Optional[float]=0.5
    profit_take_pct:Optional[float]=1.0

class VolBacktest:

    def __init__(self,data:pd.DataFrame,sizer:VolSizer,hedger:DeltaHedgeEngine,config:BacktestConfig)->None:
        self.data=data
        self.sizer=sizer
        self.hedger=hedger
        self.config=config or BacktestConfig()
        self.cost_model=TransactionalCostModel(config.costConfig)
        self.manager=PositionManager(config.ticker)

        #STates 
        self.current_pos=None
        self.shares=0.0
        self.cash=config.initial_cash

        #Accumulation of PnL attribution
        self.cum_costs=0.0
        self.hedge_pnl=0.0
        self.option_pnl=0.0


    def run(self)->None:
        
        results=[]
        prev_nav=self.config.initial_capital

        for idx,row in self.data.iterrows():

            if self.current_pos is not None:
                should_exit=self._check_stop() or (row.get('exit_signal')==True)
                if should_exit:
                    pass



    def _check_stop(self)->bool:
        if self.current_pos is None:
            return False
        
        entry_value=self.config.multiplier * self.current_pos.entry_price * self.current_pos.quantity

        if entry_value==0:
            return False
        
        pnl_pct=self.current_pos.unrealized_pnl/entry_value
        if self.config.stop_loss_pct and pnl_pct<=-self.config.stop_loss_pct:
            logger.info(f"Stop loss triggered at {pnl_pct*100:.1f}% loss.")
            return True
        if self.config.profit_take_pct and pnl_pct>=self.config.profit_take_pct:
            logger.info(f"Profit take triggered at {pnl_pct*100:.1f}% gain.")
            return True
        return False
    