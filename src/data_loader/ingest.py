import os
import io
import pandas as pd
import requests

def ingest_data():
    raw_dir = r"d:\Fpl_winning\data\raw"
    os.makedirs(raw_dir, exist_ok=True)

    base_gw_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
    base_players_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv"
    
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

    for season in seasons:
        print(f"Fetching data for {season}...")
        
        # 1. Fetch GW Data
        gw_res = requests.get(base_gw_url.format(season=season))
        if gw_res.status_code == 200:
            gw_df = pd.read_csv(io.StringIO(gw_res.text), low_memory=False)
            gw_df['season'] = season
            gw_df.to_csv(os.path.join(raw_dir, f"merged_gw_{season}.csv"), index=False)
            print(f"  -> GW data: {len(gw_df)} rows")
        else:
            print(f"  -> Failed to download GW data: HTTP {gw_res.status_code}")

        # 2. Fetch Players Raw Data (for unique Opta codes)
        p_res = requests.get(base_players_url.format(season=season))
        if p_res.status_code == 200:
            p_df = pd.read_csv(io.StringIO(p_res.text), low_memory=False)
            p_df.to_csv(os.path.join(raw_dir, f"players_raw_{season}.csv"), index=False)
            print(f"  -> Players data: {len(p_df)} rows")
        else:
            print(f"  -> Failed to download Players data: HTTP {p_res.status_code}")

if __name__ == "__main__":
    ingest_data()
