from .pricing import _d1_d2_calculate
from scipy.stats import norm
import numpy as np

def delta(S:float,K:float,T:float,r:float,sigma:float,option_type:str="call")->float:
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)
    d1,_=_d1_d2_calculate(S,K,T,r,sigma)
    if option_type=='call':
        delta_val= norm.cdf(d1)
        expired_delta=np.where(S>K,1.0,0.0)
    elif option_type=="put":
        delta_val=norm.cdf(d1)-1
        expired_delta=np.where(S>K,0.0,-1.0)
    else:
        raise ValueError("option type must be call or put")
    
    expired = (T<=0)
    return np.where(expired,expired_delta,delta_val)
    
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


# S = [100,40,30]
# K = 1
# T = 1
# r = 0.05
# sigma = 0.2

# print(delta(S,K,T,r,sigma))
# print(gamma(S,K,T,r,sigma))
# print(vega(S,K,T,r,sigma))