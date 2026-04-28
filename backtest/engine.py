import pandas as pd
import numpy as np
from strategy.sizing import VolSizer
from execution.delta_hedge import DeltaHedgeEngine
from strategy.position import PositionManager
from execution.transactions_costs import CostConfig,TransactionalCostModel
from typing import Optional
from dataclasses import dataclass,field
from backtest.portfolio import Portfolio
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

    def __init__(self,signals_df: pd.DataFrame,options_df: pd.DataFrame,sizer:VolSizer,hedger:DeltaHedgeEngine,config:BacktestConfig)->None:
        self.signals_df:pd.DataFrame=signals_df.sort_values('quote_date').reset_index(drop=True)

        self.options_df:pd.DataFrame=options_df.copy()
        options_df['quote_date']=pd.to_datetime(options_df['quote_date'])
        options_df['expiry_date']=pd.to_datetime(options_df['expiry_date'])

        self.options_df=options_df.set_index(['quote_date','strike','expiry']).sort_index()

        self.sizer:VolSizer=sizer
        self.hedger:DeltaHedgeEngine=hedger
        self.config:BacktestConfig=config or BacktestConfig()

        self.cost_model:TransactionalCostModel=TransactionalCostModel(self.config.costConfig)
        self.position_manager:PositionManager=PositionManager(ticker=self.config.ticker)
        self._portfolio:Portfolio = Portfolio(initial_capital=self.config.initial_capital, cost_config=self.config.costConfig, 
                                    multiplier=self.config.multiplier)

    def run(self)->None:
        
        results=[]
        for idx,signal_row in self.signals_df.iterrows():
            date=pd.to_datetime(signal_row['quote_date'])
            spot=float(signal_row.get('underlying_last',0.0))
            # 1. Exit Logic
            if self._portfolio.position is not None:
                should_exit = (signal_row.get('exit_signal',0)==1 or self._check_stop())
                if should_exit:
                    pos=self._portfolio.position

                    try:    
                        # Fetch today's Price for the exact contract we hold
                        contract_data = self.options_df.loc([date,pos.strike,pos.expiry])
                        contract_data['underlying_last'] = spot
                    except KeyError:
                        logger.warning(f"Data gap for Strike {pos.strike} on {date}. Forcing close using entry price.")
                        contract_data=pd.Series({"c_bid": pos.current_price/2 ,"p_bid":pos.current_price/2})
                    
                    self._portfolio.close_position(contract_data)

                    if self._portfolio.shares!=0:
                        self._portfolio.apply_hedge(-self._portfolio.shares,spot)
            
            # 2. ENtry Logic
            if self._portfolio.position is None and signal_row.get('entry_flag',0)==1:
                try:
                    #fetch all available contracts for today
                    daily_chain=self.options_df.xs(date,level='quote_date').reset_index()

                    if not daily_chain.empty():
                        best_strike=self.position_manager.select_strike(daily_chain,spot,method="delta")
                        entry_row=daily_chain[daily_chain['strike']==best_strike].iloc[0].copy()

                        entry_row['quote_date']=date
                        entry_row['underlying_last']=spot

                        qty=self.sizer.calculate_quantity(entry_row,abs(float(signal_row.get('out',1.0))))
                        new_pos=self.position_manager.create_straddle(entry_row,qty)
                        new_pos.side=int(signal_row['position'])
                        self._portfolio.open_position(new_pos,entry_row)
                except KeyError:
                    logger.warning(f"No options chain data available for entry on {date}")

            # 3. Mark to Market

            if self._portfolio.position is not None:
                pos=self._portfolio.position

                try:
                    contract_data=self.options_df.loc[(date,pos.strike,pos.expiry)].copy()
                    contract_data['underlying_last']=spot
                    snap=self._portfolio.mark_to_market(contract_data)

                except KeyError:
                    snap = self._portfolio.mark_to_market(pd.Series({"underlying_last": spot}))

            else:
                snap = self._portfolio.mark_to_market(pd.Series({"underlying_last": spot}))

            # 4 Delta Hedge
            if self._portfolio.position is not None:
                hedge_action = self.hedger.get_hedge_action(
                    current_shares=self._portfolio.shares,
                    portfolio_delta=self._portfolio.position.portfolio_delta,
                    spot=spot
                )
                self._portfolio.apply_hedge(hedge_action.shares_to_trade, spot)

            snap["date"] = date
            snap["spot"] = spot
            snap["signal"] = int(signal_row.get("signal", 0))
            snap["out"] = float(signal_row.get("out", np.nan))
            results.append(snap)

        results_df = pd.DataFrame(results)
        return self._add_drawdown(results_df)


    def _check_stop(self)->bool:
        if self._portfolio.position is None:
            return False
        
        entry_value=self.config.multiplier * self._portfolio.position.entry_price * self._portfolio.position.quantity

        if entry_value==0:
            return False
        
        pnl_pct=self._portfolio.position.unrealized_pnl/entry_value

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
