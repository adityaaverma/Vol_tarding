class DeltaHedgeEngine:
    def __init__(self,hedge_threshold:float):
        self.threshold=hedge_threshold

    def get_hedge_action(self,current_shares:float,portfolio_delta:float,quantity:float):
        """
        portfolio_delta: The Net Delta from options (e.g. Qty* (Call_Delta + Put_Delta))
        returns: the change in shares needed
        """
        net_delta=portfolio_delta * quantity
        target_shares=-net_delta

        diff=target_shares-current_shares
        if abs(diff)>self.threshold:
            return diff
        return 0.0
    
    