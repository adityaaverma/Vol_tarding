from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger=logging.getLogger(__name__)

@dataclass
class CostConfig:
    #Options
    option_commision_per_contract:float=0.06
    option_slippage_pct:float=1.00
    fallback_spread:float=0.05
    option_min_ticket:float=1.0

    #Equity
    equity_commission_per_share:float=0.005
    equity_slippage_pct:float=0.0001
    equity_min_ticket:float=1.00

    #regulatory (US - Typically updated annually)
    sec_fee_rate:float=8.0/1_000_000
    finra_taf_per_share:float=0.000145

    multiplier:float=100


# Engine

class TransactionalCostModel:
    def __init__(self,config:CostConfig|None=None):
        self.config=config or CostConfig()

    #Options
    def option_open_cost(self,row:pd.Series,quantity:float,side:int)->float:
        """
        Total cost to open straddle positions
        """ 

        config=self.config
        n=quantity

        c_spread=self._calculate_half_spread(row,"c")
        p_spread=self._calculate_half_spread(row,"p")

        spread_cost=(c_spread+p_spread)*n*config.multiplier

        #COmmission charge with minimum cost applied per leg
        c_comm=max(n*config.option_commision_per_contract,config.option_min_ticket)
        p_comm=max(n*config.option_commision_per_contract,config.option_min_ticket)
        commission=p_comm+c_comm

        c_mid=self._calculate_mid(row,"c")
        p_mid=self._calculate_mid(row,"p")
        straddle=c_mid+p_mid
        #Slippage model as percentage of premium
        slippage=straddle * config.equity_slippage_pct * n * config.multiplier

        return slippage+commission+spread_cost
    
    def option_close_cost(self,row:pd.Series,quantity:float,side:int)->float:
        """
        Total cost to close a straddle position.
        Note: If side == 1 (Long Straddle), closing means SELLING to close, triggering SEC fees.
        """
        config=self.config
        n=quantity

        c_spread=self._calculate_half_spread(row,"c")
        p_spread=self._calculate_half_spread(row,"p")

        spread_cost=(c_spread+p_spread)*n*config.multiplier

        c_comm=max(n*config.option_commision_per_contract,config.option_min_ticket)
        p_comm=max(n*config.option_commision_per_contract,config.option_min_ticket)
        commission=p_comm+c_comm

        c_mid=self._calculate_mid(row,"c")
        p_mid=self._calculate_mid(row,"p")
        straddle_mid=c_mid+p_mid

        slippage=straddle_mid * config.equity_slippage_pct * n * config.multiplier

        #Regulatory fees on closing long position
        sec_fee=0.0
        if side==1:
            sec_fee=n*config.multiplier*config.sec_fee_rate*straddle_mid   

        return slippage+commission+spread_cost+sec_fee

    #Equity
    def equity_cost(self,shares:float,spot:float)->float:
        """
        Cost for Share trade (delta hedge fill)
        Automatically handles sales detection via negative share counts
        """
        if shares==0:
            return 0.0
        
        config=self.config
        #True if selling shares (hedge or closing)        
        is_sale=shares<0

        abs_shares=abs(shares)
        commission=max(abs_shares * config.equity_commission_per_share,config.equity_min_ticket)

        slippage=abs_shares*spot*config.equity_slippage_pct

        # Regulatory fees only apply to sales
        sec_fee=config.sec_fee_rate*abs_shares*spot if is_sale else 0.0
        taf_fee=min(abs_shares*config.finra_taf_per_share,7.27) if is_sale else 0.0
        return commission + slippage + sec_fee + taf_fee
        


    def _calculate_half_spread(self,row:pd.Series,leg:str)->float:
        ask=float(row.get(f"{leg}_ask",np.nan))
        bid=float(row.get(f"{leg}_bid",np.nan))

        if np.isfinite(ask) and np.isfinite(bid) and ask>=bid:
            return (ask-bid)/2.0
        
        logger.debug(f"Missing bid/ask for {leg}. Using fallback spread.")
        return self.config.fallback_spread/2.0

    def _calculate_mid(self,row:pd.Series,leg:str)->float:
        ask=float(row.get(f"{leg}_ask",np.nan))
        bid=float(row.get(f"{leg}_bid",np.nan))

        if np.isfinite(ask) and np.isfinite(bid):
            return (ask+bid)/2.0
        
        # If no bid/ask, try last price, otherwise 0
        return float(row.get(f"{leg}_last", 0.0))
    