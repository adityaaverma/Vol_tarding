import numpy as np
import pandas as pd

class VolSizer:
    def __init__(self,target_vega_usd:float=1000):
        """
        target_vega_usd: the amount of USD we want to gain/loose if move 1pt in IV.
        """
        self.target_vega_usd=target_vega_usd

    def calculate_quantity(self,row:pd.Series)->float:
        """
        quantity=target_risk/(Call Vega + Put Vega)
        """
        total_vega=abs(row['c_vega'])+ abs(row['p_vega'])

        if total_vega==0 or np.isnan(total_vega):
            return 1.0
        
        return self.target_vega_usd/(total_vega * 100)