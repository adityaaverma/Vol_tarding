import logging
from dataclasses import dataclass
from typing import Optional,Literal

logger=logging.getLogger(__name__)


@dataclass
class HedgeConfig:
    mode:Literal['DAILY','BAND','NONE']='NONE'
    threshold:float=5.0
    hedge_cost_per_share:float=0.005
    slippage_pct:float=0.0001
    rounding:bool=True

@dataclass
class HedgeAction:
    shares_to_trade:float
    target_shares:float
    residual_delta:float
    cost:float
    reason:Optional[str]=None

    
class DeltaHedgeEngine:
    """
    Manages the neutralization od directional risk for volatility book
    """
    def __init__(self,config)->None:
        """
        Parameters:
        Hedge Threshold: float
        Minimum number of shares drift allowed before triggering a trade
        Higher threshold means low transaction costs but high directional risk
        """
        self.config=config if config else HedgeConfig()

    def calculate_hedge_action(self,current_shares:float,portfolio_delta:float,spot:float,is_closing:bool)->HedgeAction:
        """
        Computes the number of shares to remain delta neutral and cost associated with them
        """
        if is_closing==True:
            shares_to_trade=-current_shares
            cost=self._compute_hedge_cost(shares_to_trade,spot)
            logger.info(f"Closing position: trading {shares_to_trade} shares at cost {cost:.2f}")
            return HedgeAction(
                shares_to_trade=shares_to_trade,
                target_shares=0.0,
                cost=cost,
                residual_delta=0.0,
                reason="CLOSED"
            )
        
        if self.config.mode=="NONE":
            return HedgeAction(
                shares_to_trade=0.0,
                target_shares=current_shares,
                residual_delta=portfolio_delta,
                cost=0.0,
                reason="NONE_MODE"
            )
        
        target_shares=-portfolio_delta
        diff=target_shares-current_shares

        if self.config.rounding:
            diff=round(diff)
            target_shares=current_shares+diff

        shares_to_trade=0.0
        reason="WITHIN_THRESHOLD"

        if self.config.mode=='DAILY' or (self.config.mode=='BAND' and self.config.threshold<=abs(portfolio_delta)):
            shares_to_trade=diff
            reason="DAILY_REBALANCE" if self.config.mode=="DAILY" else "BAND_EXCEEDED"

        #Computed cost with slippage and commission
        cost=self._compute_hedge_cost(shares_to_trade,spot)

        #residual Delta = current Delta + Shares to Trade + Current Shares
        residual=portfolio_delta+shares_to_trade+current_shares

        if shares_to_trade != 0:
            logger.debug(f"Hedge triggered ({reason}): Trading {shares_to_trade} shares. Cost: ${cost:.2f}")

        return HedgeAction(
            shares_to_trade=shares_to_trade,
            target_shares=target_shares,
            residual_delta=residual,
            cost=cost,
            reason=reason
        )
        


    def _compute_hedge_cost(self,shares_to_trade:float,spot:float)->float:

        if shares_to_trade==0:
            return 0.0
        abs_shares=abs(shares_to_trade)
        commision=self.config.hedge_cost_per_share*abs_shares
        slippage=self.config.slippage_pct*spot*abs_shares

        return commision+slippage