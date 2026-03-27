import yfinance as yf
import pandas as pd
import logging
import datetime
import pytz
import requests

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
                (calls["strike"] >= spot) &
                (calls["strike"] >= 0.8 * spot) &
                (calls["strike"] <= 1.2 * spot)
            ]

            for _, row in calls.iterrows():
                # P = row["lastPrice"]
                P=(row['bid']+row['ask'])/2
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
                    "option_type": "call",
                    "spot": spot,
                    "iv_yf": iv_yf
                })

            # ---------- PUTS ----------
            # for out of the money puts
            puts = puts[
                # switched the logic because YF gives garbage data
                (puts["strike"] <= spot) &
                (puts["strike"] >= 0.8 * spot) &
                (puts["strike"] <= 1.2 * spot)
            ]

            for _, row in puts.iterrows():
                # P = row["lastPrice"]
                P=(row['bid']+row['ask'])/2
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
                    "option_type": "put",
                    "spot": spot,
                    "iv_yf": iv_yf
                })

        except Exception as e:
            logger.warning(f"Failed for expiry {expiry}: {e}")
            continue

    df = pd.DataFrame(data)

    return df



CBOE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def load_option_chain_cboe(symbol: str) -> pd.DataFrame:
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"

    try:
        r = requests.get(url, headers=CBOE_HEADERS, timeout=10)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        logger.error(f"CBOE fetch failed: {e}")
        return pd.DataFrame()

    spot = raw['data']['current_price']
    options = raw['data']['options']
    df_raw = pd.DataFrame(options)

    def parse_option(row):
        sym = row['option']
        # Symbol format: SPY YYMMDD C/P XXXXXXX
        date_str = sym[len(symbol):len(symbol)+6]
        exp_dt   = datetime.datetime.strptime(date_str, '%y%m%d')
        T        = (exp_dt - datetime.datetime.now()).days / 365
        opt_type = 'call' if sym[len(symbol)+6] == 'C' else 'put'
        strike   = int(sym[len(symbol)+7:]) / 1000
        return pd.Series({'T': T, 'option_type': opt_type, 'K': strike})

    parsed = df_raw.apply(parse_option, axis=1)
    df_raw = pd.concat([df_raw, parsed], axis=1)

    # Mid price
    df_raw['P'] = (df_raw['bid'] + df_raw['ask']) / 2

    # OTM only
    df_otm = pd.concat([
        df_raw[(df_raw['option_type'] == 'put')  & (df_raw['K'] <  spot)],
        df_raw[(df_raw['option_type'] == 'call') & (df_raw['K'] >= spot)]
    ])

    # Filters
    df_otm = df_otm[
        (df_otm['K']            >= 0.8 * spot) &
        (df_otm['K']            <= 1.2 * spot) &
        (df_otm['P']            >= 0.05)       &
        (df_otm['open_interest'] >= 10)         &
        (df_otm['T']            >  0)
    ]

    df_out = df_otm[['K', 'P', 'T', 'option_type', 'iv']].copy()
    df_out['spot']  = spot
    df_out['iv_yf'] = df_out['iv']
    # print(df_out)

    return df_out.sort_values(['T', 'K']).reset_index(drop=True)


