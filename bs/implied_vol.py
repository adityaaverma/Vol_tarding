from greeks import vega
from pricing import bs_price
from scipy.stats import norm
import numpy as np

def implied_vol_newton(market_price,S,K,T,r,option_type='call',initial_vol=0.2,tol=1e-8, max_iter=100):
    market_price = np.asarray(market_price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    market_price,S,K,T=np.broadcast_arrays(market_price,S,K,T)

    sigma=np.full_like(market_price,initial_vol,dtype=float)

    valid=T>0

    for _ in range(max_iter):
        price=bs_price(S,K,T,r,sigma,option_type)

        v=vega(S,K,T,r,sigma)

        diff= price-market_price
        