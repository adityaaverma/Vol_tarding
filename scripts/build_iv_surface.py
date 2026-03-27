import time
import logging
import pandas as pd
from data.loaders import load_option_chain_yahoo,load_option_chain_cboe
from vol.iv_surface import compute_iv_for_chain
from vol.interpolation import build_iv_grid
from visualization.plots import plot_iv_surface,plot_iv_smile,plot_iv_smiles

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def run_live_loop(symbol:str="SPY", refresh_seconds:float=20, n_strikes:float=80, n_maturities:float=80,r:float=0.01):
    # while True:
    now=time.time()
    logger.info("fetching option chain for %s at %s", symbol, now)
    # df=load_option_chain_cboe(symbol)
    # spot = df['spot'].iloc[0]
    # near_atm = df[(df['K'] > spot - 10) & (df['K'] < spot + 10)]
    # print("dsjbfhjevbjhrg")
    # print(near_atm[['K', 'P', 'T', 'option_type', 'iv_yf']].to_string())
    # print(df)
    df=load_option_chain_yahoo(symbol)
    if df.__len__()==0:
        return


    # if df.empty:
    #     logger.warning("no option chain data returned sleeping")
    #     time.sleep(refresh_seconds)
    #     continue

    df_iv= compute_iv_for_chain(df ,r)
    # print(df_iv)


    #     #build grid
    #     # grid_s,grid_t,grid_z=build_iv_grid(df_iv,n_strikes,n_maturities)


    # # title = f"{symbol} IV Surface {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    # # fig=plot_iv_surface(grid_s,grid_t,grid_z,title=title)

    #     # #plot smile for nearest expiry
    nearest_expiry=df_iv['T'].min()
    plot_iv_smile(df_iv[df['T']==nearest_expiry],False,nearest_expiry)
    plot_iv_smiles(df_iv,False,True)
    
    
    
    # plot_smile(df_iv,nearest_expiry,underlying=df_iv['underlying_last'].iloc[0],title=f"{symbol} IV Smile - {nearest_expiry.date()}")

        # logger.info("sleeping for %s seconds before the next update",refresh_seconds)
        # time.sleep(refresh_seconds)

if __name__=='__main__':
    run_live_loop(symbol="SPY",refresh_seconds=30)




        
