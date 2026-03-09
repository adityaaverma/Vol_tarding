import numpy as np
from scipy.stats import norm

def bs_price(S,K,T,r,sigma,option_type="call"):

    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)

    if np.any(S<=0) or np.any(K<=0):
        raise ValueError("Spot price and Strike price must be positive")

    # expiry case
    if np.any(T<=0):
        if option_type=="call":
            return np.maximum(S-K,0)
        else:
            return np.maximum(K-S,0)

    # zero vol case
    if np.any(sigma<=0):
        forward = S*np.exp(r*T)

        if option_type=="call":
            return np.exp(-r*T)*np.maximum(forward-K,0)
        else:
            return np.exp(-r*T)*np.maximum(K-forward,0)

    d1,d2 = _d1_d2_calculate(S,K,T,r,sigma)

    if option_type=="call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    elif option_type=="put":
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    

def _d1_d2_calculate(S:float,K:float,T:float,r:float,sigma:float)->tuple[float,float]:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if np.any(S<=0) or np.any(K<=0):
        raise ValueError("Spot price and strike price must be positive")
    
    d1=np.full(np.broadcast(S,K,T,sigma).shape,np.nan,dtype=float)
    d2=np.full_like(d1,np.nan)
    
    valid =(T>0) & (sigma>0)
    if np.any(valid):
        d1[valid]=(np.log(S[valid]/K[valid])+(r+0.5*sigma[valid]**2)*T[valid])/(sigma[valid]*np.sqrt(T[valid])) 
        d2[valid]=d1[valid]-sigma[valid]*np.sqrt(T[valid])

    return d1,d2

print(bs_price(100, 100, 1, 0.05, 0.2,"put"))

