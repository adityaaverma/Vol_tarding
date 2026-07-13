import pandas as pd

df = pd.read_parquet("data/SPY_ALL_YEARS_MASTER.parquet")

# Sirf wahi rows jahan DTE aur delta dono ATM straddle jaisi range mein hain
atm_sample = df[
    (df['DTE'].between(25, 55)) &
    (df['C_DELTA'].between(0.40, 0.60)) &   # ATM call ka delta ~0.5 hota hai
    (df['C_VEGA'] > 0) &
    (df['C_IV'].notna())
]

print(f"Matching rows: {len(atm_sample)}")

if len(atm_sample) > 0:
    row = atm_sample.iloc[0]
    print("Strike:", row['STRIKE'])
    print("Underlying:", row['UNDERLYING_LAST'])
    print("DTE:", row['DTE'])
    print("c_iv:", row['C_IV'])
    print("c_vega:", row['C_VEGA'])
    print("c_delta:", row['C_DELTA'])
    print("c_theta:", row['C_THETA'])
else:
    print("No matching ATM rows found — check column names/ranges")