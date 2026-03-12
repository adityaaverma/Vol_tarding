import numpy as np
from scipy.stats import norm

def bs_price(S,K,T,r,sigma,option_type="call"):

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    price = np.zeros_like(S)

    if np.any(S<=0) or np.any(K<=0):
        raise ValueError("Spot price and Strike price must be positive")
    

    expired= T<=0
    zero_vol= sigma<=0
    valid=(~expired) & (~zero_vol)

    # expiry case
    if option_type=="call":
        price[expired]=np.maximum(S[expired]-K[expired],0)
    else:
        price[expired]=np.maximum(K[expired]-S[expired],0)

    #zero vol case
    forward=S*np.exp(r*T)

    if option_type=="call":
        price[zero_vol & (~expired)]=np.exp(-r * T[zero_vol & (~expired)])* np.maximum (forward[zero_vol & (~expired)]-K[zero_vol & (~expired)],0)
    else:
        price[zero_vol & (~expired)]=np.exp(-r * T[zero_vol & (~expired)])* np.maximum (K[zero_vol & (~expired)]-forward[zero_vol & (~expired)],0)
    
    if np.any(valid):
        d1,d2 = _d1_d2_calculate(S,K,T,r,sigma)

        if option_type=="call":
            price[valid]=S[valid]*norm.cdf(d1[valid]) - K[valid]*np.exp(-r*T[valid])*norm.cdf(d2[valid])

        else:
            price[valid]=K[valid]*np.exp(-r*T[valid])*norm.cdf(-d2[valid]) - S[valid]*norm.cdf(-d1[valid])
    
    return price

def _d1_d2_calculate(S:float,K:float,T:float,r:float,sigma:float)->tuple[float,float]:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if np.any(S<=0) or np.any(K<=0):
        raise ValueError("Spot price and strike price must be positive")
    
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    d1 = np.full_like(S, np.nan, dtype=float)
    d2=np.full_like(d1,np.nan)
    
    valid =(T>0) & (sigma>0)
    if np.any(valid):
        d1[valid]=(np.log(S[valid]/K[valid])+(r+0.5*sigma[valid]**2)*T[valid])/(sigma[valid]*np.sqrt(T[valid])) 
        d2[valid]=d1[valid]-sigma[valid]*np.sqrt(T[valid])

    return d1,d2

print(bs_price([100,20,101], 100, 1, 0.05, 0.2,"put"))




