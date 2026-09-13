import os
import io
import pandas as pd
import requests

def create_dirs():
    dirs = [
        r"d:\Fpl_winning\data\raw",
        r"d:\Fpl_winning\data\processed",
        r"d:\Fpl_winning\src\data_loader"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

create_dirs()

base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

dfs = []
for season in seasons:
    print(f"Fetching {season}...")
    res = requests.get(base_url.format(season=season))
    if res.status_code == 200:
        df = pd.read_csv(io.StringIO(res.text), low_memory=False)
        df['season'] = season
        df.to_csv(rf"d:\Fpl_winning\data\raw\merged_gw_{season}.csv", index=False)
        dfs.append(df)
        print(f"  -> Downloaded {len(df)} rows. Columns: {len(df.columns)}")
        print(f"  -> Columns sample: {list(df.columns)[:5] + list(df.columns)[-5:]}")
    else:
        print(f"  -> Failed to download: HTTP {res.status_code}")
