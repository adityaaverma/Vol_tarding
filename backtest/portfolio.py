from execution.transactions_costs import CostConfig,TransactionalCostModel
from strategy.position import StraddlePosition 
import pandas as pd
import numpy as np

class Portfolio:
    """
    Single Strategy portfolio ledger

    Initial parameters
    1) Initial Capital - starting NAV (cash)
    2) cost_config - CostConfig | None
    3) multiplier - options contract multiplier (default 100)
    """

    def __init__(self,initial_capital:float=1_000_000.0,cost_config:CostConfig|None=None,multiplier:float=100):
        self.initial_captial:float=initial_capital
        self.cost_config:CostConfig=cost_config
        self.mulitplier:float=multiplier

        self._cost_model:TransactionalCostModel= TransactionalCostModel(cost_config or CostConfig())

        #Live States
        self.position:StraddlePosition | None =None
        self.shares:float=0.0

        #Cumulative Pnl Buckets (Realized)
        self._option_pnl:float=0.0
        self._hedge_pnl:float=0.0
        self._total_costs:float=0.0

        self._avg_share_cost:float=0.0
        
    def open_position(self,position:StraddlePosition,row:pd.Series)->float:
        entry_cost = self._cost_model.option_open_cost(row,position.quantity,position.side)

        c_price = row.get("c_ask", 0.0) if position.side == 1 else row.get("c_bid", 0.0)
        p_price = row.get("p_ask", 0.0) if position.side == 1 else row.get("p_bid", 0.0)

        # Fallbacks to 'mid' if raw data dropped the bid/ask
        if pd.isna(c_price) or c_price == 0: c_price = position.entry_price / 2
        if pd.isna(p_price) or p_price == 0: p_price = position.entry_price / 2

        premium_flow = -(position.side) * (c_price + p_price) * self.mulitplier * position.quantity

        self.cash +=premium_flow - entry_cost
        self._total_costs+=entry_cost
        self.position=position

        return premium_flow - entry_cost
    
    def close_position(self,row:pd.Series)->float:
        """
        Mark the position as closed using adverse fills
        """

        if self.position is None:
            return 0.0
        pos=self.position
        close_cost=self._cost_model.option_close_cost(row,pos.quantity,pos.side)

        # Adverse Fill Logic: Longs sell at Bid, Shorts buy at Ask
        c_price = row.get("c_bid", 0.0) if pos.side == 1 else row.get("c_ask", 0.0)
        p_price = row.get("p_bid", 0.0) if pos.side == 1 else row.get("p_ask", 0.0)
        
        if pd.isna(c_price) or c_price == 0: c_price = pos.current_price / 2
        if pd.isna(p_price) or p_price == 0: p_price = pos.current_price / 2

        close_flow = (pos.side) * pos.quantity * self.mulitplier * (p_price + c_price)

        self.cash+=close_flow - close_cost
        self._total_costs+=close_cost

        self._option_pnl+=pos.unrealized_pnl
        self.position=None
        
        return close_flow - close_cost

    def apply_hedge(self,share_change:float,spot:float)->float:
        """
        Execute a Delta Hedge share trade.
        """
        if share_change==0.0:
            return 0.0
        
        cost = self._cost_model.equity_cost(share_change,spot)

        cash_flow = -share_change*spot

        new_shares= self.shares + share_change

        if abs(new_shares) < 1e-9:
            # Position completely closed - Realize PnL
            realised = self.shares * (spot - self._share_avg_cost)
            self._hedge_pnl += realised
            self._share_avg_cost = 0.0
            new_shares = 0.0 # Snap to exactly zero

        elif np.sign(share_change) == np.sign(self.shares) or self.shares == 0:
            # Adding to position - Recalculate average cost basis
            total_cost_before = self.shares * self._share_avg_cost
            self._share_avg_cost = (total_cost_before + (share_change * spot)) / new_shares

        else:
            # Reducing position (but not flipping) - Realize partial PnL, Avg Cost unchanged
            realised = -share_change * (spot - self._share_avg_cost)
            self._hedge_pnl += realised

        self.shares = new_shares
        self.cash += cash_flow - cost
        self._total_costs+=cost

        return cash_flow - cost
    
    def mark_to_market(self,row:pd.Series)->dict:
        """
        Update option position with latest MID prices and return a bar snapshot.
        """
        option_pnl_change=0.0
        unrealized_opt_pnl=0.0

        if self.position is not None:
            option_pnl_change=self.position.mark_to_market(row)
            unrealized_opt_pnl=self.position.unrealized_pnl

        spot=float(row.get("underlying_last",0.0))

        unrealized_hedge_pnl= self.shares * (spot - self._avg_share_cost) if self.shares!=0.0 else 0.0

        nav = self.cash + (self.shares * self.spot) + (self.position.side *  self.position.current_price * self.position.quantity * self.mulitplier 
                                                       if self.positon else 0.0)
        
        return {
            "cash": self.cash,
            "nav": nav,
            "option_pnl_change": option_pnl_change,
            "cumulative_option_pnl": self._option_pnl + unrealized_opt_pnl,
            "cumulative_hedge_pnl": self._hedge_pnl + unrealized_hedge_pnl,
            "cumulative_costs": self._total_costs,
            "shares": self.shares,
            "has_position": self.position is not None,
        }


    
