#%%
from pathlib import Path
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from jiflr import ROOT
#%%

camps = ["8", "10", "17", "18", "25", "26", "30"]
camp_elevations = {
    "C8": 2051,
    "C10": 1190,
    "C17": 1281,
    "C18": 1703,
    "C25": 2123,
    "C26": 1447,
    "C30": 674,
}
ds = {}
for camp in camps:
    p = Path(ROOT, "data/external/mcgee_wx/JIRP Temperature Record (Main Camps).xlsx")
    ds[f"C{camp}"] = pd.read_excel(p, sheet_name=f"Camp {camp}", parse_dates=False, dtype={'Day-Month-Hour': str})

concat = []
for camp, df in ds.items():
    df = df.rename(columns={"Day-Month-Hour": "datetime"})
    df = df.drop(columns='Serial-Day')
    df = df.melt(id_vars=['datetime'], var_name="year", value_name='Ts')
    df['datetime'] = df['year'].astype(str) + " " + df['datetime'].str[5:]
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.drop(columns='year')
    df['site_id'] = camp
    df['elevation'] = camp_elevations[camp]
    df = df.dropna()
    df = df.set_index(['site_id', 'datetime'])
    concat.append(df)
    
    
#%%

ds = pd.concat(concat)
# %%

df = ds["Ts"].reset_index(level=1)
df_summer = df.loc[(df.datetime.dt.month) >= 5 & (df.datetime.dt.month <= 8)]
df_summer = df_summer.groupby(['site_id', df['datetime'].dt.year]).median()

# %%
sns.lineplot(df_summer, x='datetime', y='Ts', hue='site_id')
# %%

