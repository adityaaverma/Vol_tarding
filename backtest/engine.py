import pandas as pd
import numpy as np
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
    initial_capital:float=1_000_000.0
    costConfig:CostConfig =field(default_factory=CostConfig)
    multiplier:float=100
    ticker:str="SPY"

    #Risk Management Gaurds
    stop_loss_pct:Optional[float]=0.5
    profit_take_pct:Optional[float]=1.0

class VolBacktest:

    def __init__(self,data:pd.DataFrame,sizer:VolSizer,hedger:DeltaHedgeEngine,config:BacktestConfig)->None:
        self.data:pd.DataFrame=data
        self.sizer:VolSizer=sizer
        self.hedger:DeltaHedgeEngine=hedger
        self.config:BacktestConfig=config or BacktestConfig()

        self.cost_model:TransactionalCostModel=TransactionalCostModel(self.config.costConfig)
        self.manager:PositionManager=PositionManager(ticker=self.config.ticker)

        #States 
        self.current_pos:StraddlePosition|None=None
        self.shares:float=0.0
        self.cash:float=config.initial_capital

        #Accumulation of PnL attribution
        self.cum_costs:float=0.0
        self.hedge_pnl:float=0.0
        self.option_pnl:float=0.0


    def run(self)->None:
        
        results=[]
        prev_nav=self.config.initial_capital
        prev_spot=0.0

        for idx,row in self.data.iterrows():
            # Closing existing position if stop loss or profit take is hit or exit signal is generated
            date = row["quote_date"]
            spot=row.get('underlying_last',0)

            daily_equity_cost=0.0

            if self.current_pos is not None:
                should_exit=self._check_stop() or (row.get('exit_flag',0)==True)
                if should_exit:
                    c_price = row.get("c_bid", 0) if self.current_pos.side == 1 else row.get("c_ask", 0)
                    p_price = row.get("p_bid", 0) if self.current_pos.side == 1 else row.get("p_ask", 0)

                    #Closing all contracts and updating variables
                    proceeds=(c_price+p_price)*self.current_pos.quantity*self.config.multiplier
                    self.cash+=proceeds
                    
                    opt_cost=self.cost_model.option_close_cost(row,self.current_pos.quantity,self.current_pos.side)
                    self.cash-=opt_cost
                    self.cum_costs+=opt_cost
                    
                    if self.shares!=0:
                        # Computing Hedging costs and updating variables
                        hedge_action=self.hedger.calculate_hedge_action(self.shares,self.current_pos.delta,spot,is_closing=True)
                        self.cash-=(hedge_action.shares_to_trade*spot)

                        # Computing equity cost and updating variables
                        equity_cost=self.cost_model.equity_cost(hedge_action.shares_to_trade,spot)
                        self.cum_costs+=equity_cost
                        self.cash-=equity_cost
                        daily_equity_cost+=equity_cost

                        #No shares left
                        self.shares=0.0

                    self.current_pos=None
                    
            # Opening new position if entry signal is generated and no existing position
            if self.current_pos is None and row.get('entry_flag',0)==True:
                signal_strength = abs(float(row.get("out", 1.0)))
                quantity=self.sizer.calculate_quantity(row,signal_strength)
                self.current_pos=self.manager.create_straddle(row,quantity)

                c_price = row.get("c_ask", 0) if self.current_pos.side == 1 else row.get("c_bid", 0)
                p_price = row.get("p_ask", 0) if self.current_pos.side == 1 else row.get("p_bid", 0)

                entry_debit=(c_price+p_price)*quantity*self.config.multiplier
                self.cash-=entry_debit

                opening_cost=self.cost_model.option_open_cost(row,quantity,self.current_pos.side)
                self.cash-=opening_cost
                self.cum_costs+=opening_cost
            
            # Mark to market and Delta Hedge
            option_value=0.0    
            pos_delta=0.0
            if self.current_pos is not None:
                self.current_pos.mark_to_market(row)

                option_value=self.current_pos.side * (self.current_pos.current_price) *self.current_pos.quantity*self.config.multiplier
                pos_delta=self.current_pos.portfolio_delta
                
                if spot>0:
                    hedge_action=self.hedger.calculate_hedge_action(current_shares=self.shares,portfolio_delta=pos_delta,spot=spot,is_closing=False)

                    if hedge_action.shares_to_trade!=0:
                        self.cash-=hedge_action.shares_to_trade*spot

                        hedge_cost=self.cost_model.equity_cost(hedge_action.shares_to_trade,spot)
                        self.cash-=hedge_cost
                        self.cum_costs+=hedge_cost
                        daily_equity_cost+=hedge_cost

                        self.shares+=hedge_action.shares_to_trade

            nav=self.cash+option_value+(self.shares*spot)
            spot_change=(spot-prev_spot) if prev_spot>0 else 0.0

            # Hedge Pnl = (Shares hold * spot Move) - Equity Trading Costs
            daily_hedge_pnl=(self.shares * spot_change)-daily_equity_cost

            daily_pnl=nav-prev_nav

            daily_option_pnl=daily_pnl-daily_hedge_pnl

            #Accumulate
            self.hedge_pnl+=daily_hedge_pnl
            self.option_pnl+=daily_option_pnl

            prev_nav=nav
            prev_spot=spot

            results.append({
                "date": date,
                "nav": nav,
                "cash": self.cash,
                "daily_pnl": daily_pnl,
                "cumulative_pnl": nav - self.config.initial_capital,
                "option_pnl": self.option_pnl,       
                "hedge_pnl": self.hedge_pnl,         
                "cumulative_costs": self.cum_costs,
                "shares": self.shares,
                "has_position": 1 if self.current_pos else 0,
                "spot": spot,
                "iv": float(row.get("iv", np.nan)),
                "rv": float(row.get("rv", np.nan)),
                "spread": float(row.get("spread", np.nan)),
            })  

        results_df=pd.DataFrame(results)
        return self._add_drawdown(results_df)

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
    
    @staticmethod
    def _add_drawdown(df:pd.DataFrame)->pd.DataFrame:
        "Compute peak to trough drawdown on NAV"
        nav=df['nav'].to_numpy()
        peak=np.maximum.accumulate(nav)
        df['drawdown']=np.where(peak>0,(nav-peak)/peak,0.0)
        return df
