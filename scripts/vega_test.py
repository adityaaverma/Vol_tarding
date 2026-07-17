import pandas as pd

df = pd.read_parquet("data/SPY_ALL_YEARS_MASTER.parquet")

# Sirf wahi rows jahan DTE aur delta dono ATM straddle jaisi range mein hain
atm_sample = df[
    (df['dte'].between(25, 55)) &
    (df['c_delta'].between(0.40, 0.60)) &   # ATM call ka delta ~0.5 hota hai
    (df['c_vega'] > 0) &
    (df['c_iv'].notna())
]

print(f"Matching rows: {len(atm_sample)}")

if len(atm_sample) > 0:
    row = atm_sample.iloc[0]
    print("Strike:", row['strike'])
    print("Underlying:", row['underlying_last'])
    print("DTE:", row['dte'])
    print("c_iv:", row['c_iv'])
    print("c_vega:", row['c_vega'])
    print("c_delta:", row['c_delta'])
    print("c_theta:", row['c_theta'])
else:
    print("No matching ATM rows found — check column names/ranges")