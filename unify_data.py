import pandas as pd
import glob
import os

files = glob.glob(r"d:\Fpl_winning\data\raw\merged_gw_*.csv")
dfs = []

# Target columns essential for the AI Optimizer Implementation Plan
core_cols = [
    'season', 'name', 'position', 'team', 'GW', 
    'element', # player id
    'fixture', 'opponent_team', 'was_home',
    'total_points', 'value', 'minutes', 
    'goals_scored', 'assists', 'clean_sheets', 
    'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed', 
    'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps',
    'influence', 'creativity', 'threat', 'ict_index',
    'transfers_balance', 'selected', 'transfers_in', 'transfers_out',
    'xP', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded'
]

for f in files:
    df = pd.read_csv(f, low_memory=False)
    
    # Map gameweek / round to GW uniquely
    if 'GW' not in df.columns:
        if 'gameweek' in df.columns:
            df.rename(columns={'gameweek': 'GW'}, inplace=True)
        elif 'round' in df.columns:
            df.rename(columns={'round': 'GW'}, inplace=True)
            
    # Map player_id / id to element
    if 'element' not in df.columns:
        if 'player_id' in df.columns:
            df.rename(columns={'player_id': 'element'}, inplace=True)
        elif 'id' in df.columns:
            df.rename(columns={'id': 'element'}, inplace=True)
            
    # Map price to value
    if 'value' not in df.columns and 'price' in df.columns:
        df.rename(columns={'price': 'value'}, inplace=True)
        
    # Ensure all core columns exist; fill with NaN if an older season lacked it (e.g. expected_goals)
    for col in core_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Deal with duplicate columns if they accidentally exist
    df = df.loc[:, ~df.columns.duplicated()]

    # Keep only the unified core columns
    df = df[core_cols]
    dfs.append(df)

unified_df = pd.concat(dfs, ignore_index=True)

# Sort strictly by player, season, and gameweek for time-series integrity
unified_df.sort_values(by=['element', 'season', 'GW'], inplace=True)

# Save to processed directory
unified_df.to_parquet(r"d:\Fpl_winning\data\processed\unified_fpl_data.parquet", index=False)
unified_df.to_csv(r"d:\Fpl_winning\data\processed\unified_fpl_data.csv", index=False)

print("--- Unified Dataset Summary ---")
print(f"Total Rows: {len(unified_df):,}")
print(f"Total Columns: {len(unified_df.columns)}")
print(f"Seasons Included: {unified_df['season'].unique().tolist()}")
print(f"Total Unique Players (Elements): {unified_df['element'].nunique()}")
print(f"\nSchema / Columns:")
print(list(unified_df.columns))
