import pandas as pd
import glob
import os
import numpy as np

def clean_data():
    raw_dir = r"d:\Fpl_winning\data\raw"
    processed_dir = r"d:\Fpl_winning\data\processed"
    os.makedirs(processed_dir, exist_ok=True)

    gw_files = glob.glob(os.path.join(raw_dir, "merged_gw_*.csv"))
    
    dfs = []
    
    # Target columns
    core_cols = [
        'season', 'player_code', 'element', 'name', 'position', 'team', 'GW', 
        'fixture', 'opponent_team', 'was_home',
        'total_points', 'value', 'minutes', 
        'goals_scored', 'assists', 'clean_sheets', 
        'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed', 
        'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps',
        'influence', 'creativity', 'threat', 'ict_index',
        'transfers_balance', 'selected', 'transfers_in', 'transfers_out',
        'xP', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded'
    ]

    for gw_file in gw_files:
        season = gw_file.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(gw_file, low_memory=False)
        
        # Mapping column names safely
        rename_map = {
            'gameweek': 'GW',
            'round': 'GW',
            'player_id': 'element',
            'id': 'element',
            'price': 'value'
        }
        
        # Only rename if target doesn't exist to prevent duplicates
        for old_col, new_col in rename_map.items():
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # Merge with players_raw to get the unique `code`
        players_file = os.path.join(raw_dir, f"players_raw_{season}.csv")
        if os.path.exists(players_file):
            players_df = pd.read_csv(players_file, low_memory=False)
            # players_raw 'id' matches GW 'element'
            # we need 'code' which is the unique Opta ID
            player_map = players_df[['id', 'code']].drop_duplicates()
            df = df.merge(player_map, left_on='element', right_on='id', how='left')
            df.rename(columns={'code': 'player_code'}, inplace=True)
            if 'id' in df.columns:
                df.drop(columns=['id'], inplace=True)
        else:
            df['player_code'] = np.nan

        # Ensure all core columns exist
        for col in core_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df.loc[:, ~df.columns.duplicated()]
        df = df[core_cols]
        dfs.append(df)

    if dfs:
        unified_df = pd.concat(dfs, ignore_index=True)
        
        # Type conversions and basic cleaning
        unified_df['GW'] = pd.to_numeric(unified_df['GW'], errors='coerce')
        unified_df['player_code'] = pd.to_numeric(unified_df['player_code'], errors='coerce')
        unified_df['minutes'] = pd.to_numeric(unified_df['minutes'], errors='coerce')
        
        # Drop corrupted rows without valid GW or player code
        unified_df.dropna(subset=['GW', 'player_code'], inplace=True)
        
        # Handle duplicate player/GW rows (e.g. Double Gameweeks might have 2 rows, we need to aggregate them!)
        # Actually, for point-in-time features, DGWs mean a player plays 2 matches in 1 GW.
        # We need to sum up their points/minutes, but keep the GW identifier intact.
        
        agg_funcs = {col: 'first' for col in core_cols}
        
        # Columns to sum for DGWs
        sum_cols = ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
                    'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed',
                    'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps']
        
        # Convert expected stats to numeric for aggregation
        expected_cols = ['xP', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']
        for c in expected_cols:
            unified_df[c] = pd.to_numeric(unified_df[c], errors='coerce')
            sum_cols.append(c)
            
        for col in sum_cols:
            if col in agg_funcs:
                agg_funcs[col] = 'sum'
                
        # Groupby player_code, season, and GW to handle Double Gameweeks properly
        unified_df = unified_df.groupby(['player_code', 'season', 'GW'], as_index=False).agg(agg_funcs)

        unified_df.sort_values(by=['player_code', 'season', 'GW'], inplace=True)
        
        unified_df.to_parquet(os.path.join(processed_dir, "cleaned_fpl_data.parquet"), index=False)
        print(f"Data cleaned and saved: {len(unified_df)} unique player-gameweek observations.")

if __name__ == "__main__":
    clean_data()
