import yfinance as yf
import pandas as pd
import logging
import datetime
import pytz

logger = logging.getLogger(__name__)

def load_option_chain_yahoo(symbol: str) -> pd.DataFrame:
    '''
    Fetched 
    Spot Price, Strike, expiries, lastPrice(price of the option at which it was last traded), 
    implied volatitlity for the respective option contract and created a dataframe
    
    '''
    # used NYC time because yf fetches option contract in NYC time
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.datetime.now(ny_tz)

    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options[7:15])

    data = []

    spot = ticker.fast_info["last_price"]

    for expiry in expiries:
        try:
            expiry_dt = pd.to_datetime(expiry).tz_localize(ny_tz)

            T = (expiry_dt - now).total_seconds() / (365 * 24 * 3600)

            if T <= 0:
                continue

            chain = ticker.option_chain(expiry)

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # ---------- CALLS ----------
            # for out of the money calls
            calls = calls[
                # switched the logic because YF gives garbage data
                (calls["strike"] <= spot) &
                (calls["strike"] >= 0.8 * spot) &
                (calls["strike"] <= 1.2 * spot)
            ]

            for _, row in calls.iterrows():
                P = row["lastPrice"]
                K = row["strike"]
                iv_yf = row["impliedVolatility"]

                if (
                    P < 0.05 or
                    row["openInterest"] < 10
                ):
                    continue

                data.append({
                    "K": K,
                    "P": P,
                    "T": T,
                    "option_type": "call" if K>=spot else "put",
                    "spot": spot,
                    "iv_yf": iv_yf
                })

            # ---------- PUTS ----------
            # for out of the money puts
            puts = puts[
                # switched the logic because YF gives garbage data
                (puts["strike"] > spot) &
                (puts["strike"] >= 0.8 * spot) &
                (puts["strike"] <= 1.2 * spot)
            ]

            for _, row in puts.iterrows():
                P = row["lastPrice"]
                K = row["strike"]
                iv_yf = row["impliedVolatility"]

                if (
                    P < 0.05 or
                    row["openInterest"] < 10
                ):
                    continue

                data.append({
                    "K": K,
                    "P": P,
                    "T": T,
                    "option_type": "call" if K>=spot else "put",
                    "spot": spot,
                    "iv_yf": iv_yf
                })

        except Exception as e:
            logger.warning(f"Failed for expiry {expiry}: {e}")
            continue

    df = pd.DataFrame(data)

    return df