import pandas as pd
import numpy as np
import os

def build_features():
    print("Loading cleaned data...")
    df = pd.read_parquet(r"d:\Fpl_winning\data\processed\cleaned_fpl_data.parquet")
    
    # 1. Target Variables for current GW (what we want to predict)
    df['target_points'] = df['total_points']
    df['target_minutes'] = df['minutes']
    df['target_played'] = (df['minutes'] > 0).astype(int)
    
    # Re-insert NaN for expected stats in 2021-22 where they were unavailable (summed to 0)
    expected_cols = ['xP', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']
    for col in expected_cols:
        df.loc[(df['season'] == '2021-22') & (df[col] == 0), col] = np.nan
        
    print("Sorting chronologically...")
    df.sort_values(by=['player_code', 'season', 'GW'], inplace=True)
    
    # Base columns to shift (T-1)
    shift_cols = ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 
                  'goals_conceded', 'bonus', 'bps', 'value', 'selected', 'transfers_balance'] + expected_cols
                  
    print("Shifting features to enforce T-1 point-in-time...")
    for col in shift_cols:
        df[f'{col}_last_1'] = df.groupby('player_code')[col].shift(1)
        
    # Rolling averages over the last 3, 5, 10 games
    windows = [3, 5, 10]
    print("Computing rolling features...")
    for w in windows:
        # Use min_periods=1 so we don't drop players with < w games
        df[f'pts_last_{w}'] = df.groupby('player_code')['total_points_last_1'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'mins_last_{w}'] = df.groupby('player_code')['minutes_last_1'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'xg_last_{w}'] = df.groupby('player_code')['expected_goals_last_1'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'xa_last_{w}'] = df.groupby('player_code')['expected_assists_last_1'].transform(lambda x: x.rolling(w, min_periods=1).mean())
        
    # Season-to-date stats (Expanding mean within the same season)
    print("Computing season-to-date features...")
    df['pts_std'] = df.groupby(['player_code', 'season'])['total_points_last_1'].transform(lambda x: x.expanding().mean())
    df['mins_std'] = df.groupby(['player_code', 'season'])['minutes_last_1'].transform(lambda x: x.expanding().mean())
    df['xg_std'] = df.groupby(['player_code', 'season'])['expected_goals_last_1'].transform(lambda x: x.expanding().mean())
    
    # Future Leakage Prevention: Remove rows where we don't have a valid target (e.g., if we shift and miss)
    # Actually, we keep all rows, but we drop NA targets during training.
    
    # Save the feature dataset
    out_path = r"d:\Fpl_winning\data\processed\features_targets.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Feature engineering complete. Saved to {out_path}")
    print(f"Total columns: {len(df.columns)}")

if __name__ == "__main__":
    build_features()
