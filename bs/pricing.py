import numpy as np
from scipy.stats import norm

def bs_price(S:float,K:float,T:float,r:float, sigma:float, option_type:str="call")->float:
    """
    Black sholes call option price

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk free rate
    sigma: volatility
    """
    # 
    if S<=0 or K<=0:
        raise ValueError("Spot price and Strike price must be positive")
    
    #handling sigma <=0
    if sigma <=0:
        forward = S* np.exp(r*T)
        if option_type=="call":
            return np.exp(-r*T)*max(forward-K,0)
        elif option_type=="put":
            return np.exp(-r*T)*max(K-forward,0)
        
    # handling T<=0 
    if T<=0:
        if option_type=="call":
            return max(S-K,0)
        elif option_type=="put":
            return max(K-S,0)
        
    d1,d2=_d1_d2_calculate(S,K,T,r,sigma)
    if option_type=="call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type=="put":
        return  K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1) 
    else:
        raise ValueError("option type must be either call or put")
    

def _d1_d2_calculate(S:float,K:float,T:float,r:float,sigma:float)->tuple[float,float]:
    d2=((np.log(S)-np.log(K))+(r-((sigma**2)/2))*T)/(sigma*np.sqrt(T))
    d1=d2+sigma*np.sqrt(T)
    return d1,d2

print(bs_price(100, 100, 1, 0.05, 0.2,"put"))

