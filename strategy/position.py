from dataclasses import dataclass,field
import pandas as pd
from typing import Dict

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
    quantiy:float
    entry_price:float
    entry_spot:float

    #Live Metrics
    current_price:float=field(init=False)
    current_spot:float=field(init=False)
    unrealized_pnl:float=field(init=False)

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
        self.attribution['delta_pnl']+=self.side * self.delta * ds * self.quantity * LOT_SIZE
        self.attribution['gamma_pnl']+=self.side * 0.5 * self.gamma * (ds**2) * self.quantity * LOT_SIZE
        self.attribution['theta_pnl']+=self.side * self.theta * dt * self.quantiy * LOT_SIZE

        #update state
        self.current_price=new_price
        self.current_spot=new_spot
        self.unrealized_pnl+=total_pnl_change

        #Refresh greeks for next bars attibution
        self.delta=float(row.get('c_delta',0.0)+row.get('p_delta',0.0))
        self.vega=float(row.get('c_vega',0.0)+row.get('p_vega',0.0))
        self.theta=float(row.get('c_theta',0.0)+row.get('p_theta',0.0))
        self.gamma=float(row.get('c_gamma',0.0)+row.get('p_gamma',0.0))

        return total_pnl_change
    
    @property
    def portfolio_delta(self)->float:
        return self.delta * self.quantity * LOT_SIZE
    
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

        return StraddlePosition(
            ticker=self.ticker,
            entry_date=row['quote_date'],
            expiry=row['expire_date'],
            strike=row['strike'],
            side=row['position'],
            quantiy=quantity,
            entry_price=c_price+p_price,
            entry_spot=row['underlying_last']
        )