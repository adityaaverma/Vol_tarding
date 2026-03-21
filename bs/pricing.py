import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option_type="call"):

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    option_type = np.asarray(option_type)

    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    option_type = np.broadcast_to(option_type, S.shape)

    price = np.zeros_like(S)

    if np.min(S) <= 0 or np.min(K) <= 0:
        raise ValueError("Spot price and Strike price must be positive")

    is_call = option_type == 'call'
    expired = T <= 0
    zero_vol = sigma <= 0
    valid = (~expired) & (~zero_vol)

    # Expiry
    price[expired] = np.where(
        is_call[expired],
        np.maximum(S[expired] - K[expired], 0),
        np.maximum(K[expired] - S[expired], 0)
    )

    # Zero vol
    mask = zero_vol & (~expired)
    if np.any(mask):
        forward = S[mask] * np.exp(r * T[mask])
        price[mask] = np.where(
            is_call[mask],
            np.exp(-r * T[mask]) * np.maximum(forward - K[mask], 0),
            np.exp(-r * T[mask]) * np.maximum(K[mask] - forward, 0)
        )

    # Valid case
    if np.any(valid):
        Sv = S[valid]
        Kv = K[valid]
        Tv = T[valid]
        sigv = sigma[valid]

        d1, d2 = _d1_d2_calculate(Sv, Kv, Tv, r, sigv)

        price[valid] = np.where(
            is_call[valid],
            Sv * norm.cdf(d1) - Kv * np.exp(-r * Tv) * norm.cdf(d2),
            Kv * np.exp(-r * Tv) * norm.cdf(-d2) - Sv * norm.cdf(-d1)
        )

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

# print(bs_price([100,20,101], 100, 1, 0.05, 0.2,"put"))




