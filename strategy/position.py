import numpy as np
from dataclasses import dataclass
import pandas as pd

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


class PositionManager:
    """
     handles the logic of selecting right strike and defining straddle attributes
    """

    def __init__(self,ticker="SPY"):
        self.ticker=ticker

    def create_straddle(self,row:pr.Series,quantity:float)->StraddlePosition:
        """
        Uses the rows from Signal.Rules output to define a trade
        """
        c_price=(row['c_bid']+row['c_ask'])/2 if 'c_bid' in row and 'c_ask' in row else 0
        b_price=(row['p_bid']+row['p_ask'])/2 if 'p_bid' in row and 'p_ask' in row else 0   

        return StraddlePosition(
            ticker=self.ticker,
            entry_date=row['quote_date'],
            expiry=row['expire_date'],
            strike=row['strike'],
            side=row['position'],
            quantiy=quantity,
            entry_price=c_price+b_price,
            entry_spot=row['spot']
        )