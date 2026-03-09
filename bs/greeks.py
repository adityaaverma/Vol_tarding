from bs.pricing import _d1_d2_calculate
from scipy.stats import norm
import numpy as np

def delta(S:float,K:float,T:float,r:float,sigma:float,option_type:str="call")->float:
    d1,_=_d1_d2_calculate(S,K,T,r,sigma)
    if option_type=='call':
        return norm.cdf(d1)
    elif option_type=="put":
        return norm.cdf(d1)-1
    else:
        raise ValueError("option type must be call or put")
    
def gamma(S:float,K:float,T:float,r:float,sigma:float)->float:
    S = np.asarray(S)
    T = np.asarray(T)
    sigma = np.asarray(sigma)

    d1,_=_d1_d2_calculate(S,K,T,r,sigma)
    gamma_val= norm.pdf(d1)/(S*sigma*np.sqrt(T))
    invalid= np.isnan(d1) | (sigma<=0) | (T<=0)
    return np.where(invalid,0.0,gamma_val)

def vega(S:float,K:float,T:float,r:float,sigma:float)->float:
    S = np.asarray(S)
    T = np.asarray(T)
    sigma = np.asarray(sigma)

    d1,_=_d1_d2_calculate(S,K,T,r,sigma)
    vega_val= S*norm.pdf(d1)*np.sqrt(T)
    invalid=np.isnan(d1)|(sigma<=0)|(T<=0)
    return np.where(invalid,0.0,vega_val)

    