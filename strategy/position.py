from dataclasses import dataclass,field
from email.policy import default
import pandas as pd
from typing import Dict
import numpy as np

LOT_SIZE=100

def _mid(row:pd.Series,side:str)->float:
    bid,ask=row.get(f"{side}_bid"),row.get(f"{side}_ask")
    if pd.isna(bid) or pd.isna(ask):
        return float(row.get(f"{side}_last",0.0))
    return (bid+ask)/2

@dataclass
class StraddlePosition:
    ticker:str
    entry_date:pd.Timestamp
    expiry:pd.Timestamp
    strike:float
    side:int # 1 for long, -1 for short
    quantity:float
    entry_price:float
    entry_spot:float

    #Live Metrics
    current_price:float=field(init=False)
    current_spot:float=field(init=False)
    current_iv:float=field(init=False)
    unrealized_pnl:float=field(default=0.0)

    #Greeks
    delta:float=0.0
    vega:float=0.0
    theta:float=0.0
    gamma:float=0.0

    attribution:Dict[str,float]=field(default_factory=lambda:{
        "delta_pnl":0.0,
        "gamma_pnl":0.0,
        "vega_pnl":0.0,
        "theta_pnl":0.0,
    })  

    def __post_init__(self)->None:
        if self.side not in [1,-1]:
            raise ValueError(f"side must be 1 (long) or -1 (short) got {self.side}")
        if self.quantity<=0:
            raise ValueError(f"quantity must be positive got {self.quantity}")
        self.current_price=self.entry_price
        self.current_spot=self.entry_spot
        self.unrealized_pnl=0.0
    
    def _safe_num(row: pd.Series, key: str, default: float = 0.0) -> float:
        val = row.get(key, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    def mark_to_market(self,row:pd.Series)->float:
        """
        update position and performs first order pnl attribution
        """
        new_price=_mid(row,'c')+_mid(row,'p')
        new_spot=row.get('underlying_last',self.current_spot)

        #total Pnl Chnage
        total_pnl_change=self.side * (new_price - self.current_price) * self.quantity * LOT_SIZE

        #using taylor series expansion on portfolio value change to attribute pnl to greeks
        # delta_pnl=Delta*dS + 0.5*Gamma*ds^2 + Vega*dIv + Theta*dt

        ds=new_spot - self.current_spot
        dt=1/252 # assuming daily steps

        # We calculate attribution to check if our greeks explain price movement 
        delta_chg=self.side * self.delta * ds * self.quantity * LOT_SIZE
        gamma_chg=self.side * 0.5 * self.gamma * (ds**2) * self.quantity * LOT_SIZE
        theta_chg=self.side * self.theta * self.quantity * LOT_SIZE
       

        c_iv = float(row.get('c_iv', np.nan))
        p_iv = float(row.get('p_iv', np.nan))
        if np.isfinite(c_iv) and np.isfinite(p_iv):
            new_iv = (c_iv + p_iv) / 2.0
        elif np.isfinite(c_iv):
            new_iv = c_iv
        elif np.isfinite(p_iv):
            new_iv = p_iv
        else:
            new_iv = self.current_iv   # no IV data → no vega attribution this bar
 
        div = new_iv - self.current_iv  # change in implied vol (annualised decimal)
        vega_chg = self.side * (self.vega * 100) * div * self.quantity * LOT_SIZE

        self.attribution['delta_pnl'] += delta_chg
        self.attribution['gamma_pnl'] += gamma_chg
        self.attribution['theta_pnl'] += theta_chg
        self.attribution['vega_pnl']  += vega_chg 


        #update state
        self.current_price=new_price
        self.current_spot=new_spot
        self.unrealized_pnl+=total_pnl_change
        self.current_iv=new_iv

        #Refresh greeks for next bars attibution
        self.delta = self._safe_num(row,'c_delta') + self._safe_num(row,'p_delta')
        self.vega  = self._safe_num(row,'c_vega')  + self._safe_num(row,'p_vega')
        self.theta = self._safe_num(row,'c_theta') + self._safe_num(row,'p_theta')
        self.gamma = self._safe_num(row,'c_gamma') + self._safe_num(row,'p_gamma')

        return total_pnl_change
    
    @property
    def portfolio_delta(self)->float:
        return self.side * self.delta * self.quantity * LOT_SIZE
    
class PositionManager:
    """
     handles the logic of selecting right strike and defining straddle attributes
    """

    def __init__(self,ticker="SPY"):
        self.ticker=ticker

    def select_strike(self,options_df:pd.Series,spot:float,method:str="delta")->float:
        if options_df.empty:
            return None
        if method=='delta' and 'c_delta' in options_df.columns:
            idx=(options_df['c_delta']-0.5).abs().idxmin()
        else:
            idx=(options_df['strike']-spot).abs().idxmin()
        return float(options_df.loc[idx,'strike'])
    
    def create_straddle(self,row:pd.Series,quantity:float)->StraddlePosition:
        """
        Uses the rows from Signal.Rules output to define a trade
        """
        c_price=_mid(row,'c')
        p_price=_mid(row,'p')

        if c_price==0 and p_price==0:
            raise ValueError(f"Zero price detected for {self.ticker} on {row['quote_date']}. Data gap?")   
        
        side=int(row.get("position",0))
        if side not in [1,-1]:
            raise ValueError(f"row['position'] must be 1 or -1 before calling create_straddle, got {side}. "
            "Set entry_row['position'] = int(signal_row['position']) before this call."
            )

        pos:StraddlePosition= StraddlePosition(
            ticker=self.ticker,
            entry_date=row.get('quote_date',0),
            expiry=row.get('expire_date',0),
            strike=row.get('strike',0),
            side=row.get('position',0),
            quantity=quantity,
            entry_price=c_price+p_price,
            entry_spot=row.get('underlying_last',0)
        )
        c_iv=float(row.get('c_iv',np.nan))
        p_iv=float(row.get('p_iv',np.nan))

        if np.isfinite(c_iv) and np.isfinite(p_iv):
            entry_iv=(p_iv+c_iv)/2.0
        elif np.isfinite(p_iv):
            entry_iv=p_iv
        elif np.isfinite(c_iv):
            entry_iv=c_iv
        else:
            entry_iv=0.0
        pos.current_iv=entry_iv
        return pos