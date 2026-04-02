import numpy as np
from vol.metrics import iv_rv_spread,z_score

class volSignalEngine:
    def __init__(self,config):
        self.window=config.get('window', 20)
        pass

    def compute_features(self,data):
        iv=data['iv']
        rv=data['rv']
        fwd_rv=data['fwd_rv']

        spread=iv_rv_spread(iv,rv)
        fwd_spread=iv_rv_spread(iv,fwd_rv)

        z=z_score(spread,self.window)

        return {
            "iv": iv,
            "rv": rv,
            "fwd_rv": fwd_rv,
            "spread": spread,
            "fwd_spread": fwd_spread,
            "z_score": z
        }
    
    

    