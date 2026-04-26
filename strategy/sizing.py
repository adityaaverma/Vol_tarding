import numpy as np
import pandas as pd
import logging 
from dataclasses import dataclass
from typing import Literal, Optional

logger=logging.getLogger(__name__)


@dataclass
class SizerConfig:
    mode:Literal['vega','vol','kelly']='vega'

    #Vega mode (Risk based)
    target_vega_usd:float=1_000.0

    #Vol target Mode (portfolio based)
    target_vol:float=0.15       #15% annulaized volatility target
    portfolio_value:float=1_000_000.0

    # Kelly Mode (Signal Based)
    kelly_fraction:float=0.20  #Use kelly fraction to avoid over betting
    win_rate:float=0.52 # Emperical Win rate
    avg_win_loss_ratio:float=1.2 #Profit factor

    #Universal risk gaurds
    max_contracts:float=100.0
    min_contracts:float=1.0
    max_notional_pct:float=0.10  #Max of 10% percent of portfolio value per position
    multiplier:int =100  #Standard equity option multiplier

class VolSizer:
    def __init__(self,config:Optional[SizerConfig]=None)->None:
        self.config=config or SizerConfig()

    def calculate_quantity(self,row:pd.Series,signal_strength:float=1.0)->float:
        """
        calculate quantity based on the selected mode with safety overrides
        """

        cfg=self.config
        if cfg.mode=='vega':
            qty=self._vega_size(row)
        elif cfg.mode=='vol':
            qty=self._vol_target_size(row)
        elif cfg.mode=='kelly':
            qty=self._kelly_size(signal_strength)
        else:
            raise ValueError(f"unsupported sizing mode: {cfg.mode}")
        
        return qty
        

    def _vega_size(self,row:pd.Series)->float:
        """
        Calculates quantity dollar exposure to IV constant
        Qty:target_USD_vega/(contarct_vega * multiplier)
        """

        total_vega=abs(float(row.get('c_vega',0)))+abs(float(row.get('p_vega',0)))

        if total_vega<0.001:
            logger.warning("vega is missing or zero. Sizing at minimum.")
            return self.config.min_contracts
        
        return self.config.target_vega_usd/(total_vega*self.config.multiplier)
        
    def _vol_target_size(self,row:pd.Series)->float:
        """
        Targets a specfic daily dollar volatility 
        Intution: Options risk is driven by Vega and Gamma. We use Vega proxy here
        """
        cfg=self.config
        iv=float(row.get('iv',0))

        total_vega_usd=(abs(float(row.get('c_vega',0)))+ abs(float(row.get('p_vega',0))))* cfg.multiplier

        if iv<=0 or total_vega_usd<=0:
            return cfg.min_contracts
        
        # Daily dollar vol target = (portfolio value * annula_target_vol)/sqrt(252)
        target_daily_dollars=cfg.portfolio_value*cfg.target_vol/np.sqrt(252)

        # Contract daily vol approximation
        # based on relation as dPnl approx= vega*dVol
        contract_daily_vol=total_vega_usd*(iv/np.sqrt(252))

        return target_daily_dollars/contract_daily_vol
    
    def _kelly_size(self,signal_strength:float)->float:
        """
        sizes based on kelly criterion: f*=(bp-q)/b
        scale based by 'signal strength' (z_score) 
        """
        cfg=self.config
        p=cfg.win_rate
        q=1-p
        b=cfg.avg_win_loss_ratio

        full_kelly=(p*b-q)/b
        fraction_kelly=max(0,full_kelly) * cfg.kelly_fraction * np.tanh(abs(signal_strength))

        return cfg.max_contracts*fraction_kelly       
        

    def _get_straddle_premium(row:pd.Series)->float:
        c_mid=(row.get('c_bid',0))+(row.get('c_ask',0))/2
        p_mid=(row.get('p_bid',0))+(row.get('p_ask',0))/2
        return c_mid+p_mid