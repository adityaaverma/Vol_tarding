from .greeks import vega
from .pricing import bs_price
import numpy as np

def implied_vol_newton(market_price,S,K,T,r,option_type='call',initial_vol=0.2,tol=1e-8, max_iter=100):
    market_price = np.asarray(market_price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    option_type = np.asarray(option_type,dtype=str)

    market_price,S,K,T=np.broadcast_arrays(market_price,S,K,T)

    market_price = np.atleast_1d(market_price)
    S = np.atleast_1d(S)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)
    option_type=np.broadcast_to(option_type,market_price.shape)

    #arbitrage check
    is_call=option_type=='call'

    disc=np.exp(-r*T)

    lower_bound=np.where(is_call,np.maximum(S-K*disc,0),np.maximum(K*disc-S,0))
    upper_bound=np.where(is_call,S,K*disc)
    # if option_type=='call':
    #     lower_bound=np.maximum(S-K*disc,0)
    #     upper_bound=S
    # else: 
    #     lower_bound=np.maximum(K*disc-S,0)
    #     upper_bound=K*disc

    invalid=(market_price<lower_bound) | (market_price>upper_bound)
    
    sigma=np.full_like(market_price,initial_vol,dtype=float)

    active=(T>0) & (~invalid)

    for _ in range(max_iter):
        if not np.any(active):
            break
        price=bs_price(S[active],K[active],T[active],r,sigma[active],option_type[active])

        v=vega(S[active],K[active],T[active],r,sigma[active])

        diff= price-market_price[active]
        converged=np.abs(diff)<tol

        update=np.zeros_like(v)
        safe=v>1e-12 

        update[safe]= diff[safe]/v[safe]
        active_idx=np.where(active)[0]
        safe_idx=active_idx[safe]
        sigma[safe_idx]-=update[safe]
        sigma=np.clip(sigma,1e-6,5.0)
        active[active_idx[converged]]=False

    sigma[invalid]=np.nan
    return sigma


def implied_vol_bisection(market_price,S,K,T,r,option_type='call',low=1e-6,high=5.0,tol=1e-8,max_iter=200):


    market_price = np.asarray(market_price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    option_type = np.asarray(option_type,dtype=str)

    market_price, S, K, T = np.broadcast_arrays(market_price, S, K, T)
    option_type=np.broadcast_to(option_type,market_price.shape)

    low_vol=np.full_like(S,low,dtype=float)
    high_vol=np.full_like(S,high,dtype=float)

    for _ in range(max_iter):
        mid=0.5*(low_vol+high_vol)
        price=bs_price(S,K,T,r,mid,option_type)

        diff=price-market_price
        high_vol[diff>0]=mid[diff>0]
        low_vol[diff<=0]=mid[diff<=0]

        if np.all(np.abs(diff)<tol):
            break
    return 0.5*(low_vol+high_vol)

def implied_vol(market_price,S,K,T,r,option_type='call'):
    sigma=implied_vol_newton(market_price,S,K,T,r,option_type)

    price=bs_price(S,K,T,r,sigma,option_type)
    diff=np.abs(price-market_price)

    failed=diff>1e-6
    if np.any(failed):
        sigma[failed]=implied_vol_bisection(market_price[failed],S[failed],K[failed],T[failed],r,option_type[failed])

    return sigma


# S = 100
# K = 100
# T = 1
# r = 0.05
# true_vol = 0.2

# price = bs_price(S, K, T, r, true_vol)

# iv = implied_vol(price, S, K, T, r)

# print(iv)