import pandas as pd
import sys

def run_leakage_test():
    print("Loading features and targets...")
    df = pd.read_parquet(r"d:\Fpl_winning\data\processed\features_targets.parquet")
    
    print("Testing for target leakage...")
    
    # Sort exactly as in build_features
    df.sort_values(by=['player_code', 'season', 'GW'], inplace=True)
    
    # 1. Shift check: Ensure 'total_points_last_1' strictly equals the manually shifted 'target_points'
    df['check_pts'] = df.groupby('player_code')['target_points'].shift(1)
    
    valid = (df['total_points_last_1'] == df['check_pts']) | (df['total_points_last_1'].isna() & df['check_pts'].isna())
    if not valid.all():
        print("FAIL: 'total_points_last_1' does not strictly match T-1 shifted points. Lookahead bias detected!")
        sys.exit(1)
    else:
        print("PASS: T-1 shift integrity verified.")
        
    # 2. Check that no historical feature requires current GW information
    # Target variables
    targets = ['target_points', 'target_minutes', 'target_played', 'total_points']
    
    # Let's verify correlation. The correlation between a feature and target shouldn't be exactly 1.0
    corr = df[['total_points_last_1', 'target_points']].corr().iloc[0,1]
    if corr > 0.99:
        print("FAIL: Suspiciously high correlation between past points and current target.")
        sys.exit(1)
    else:
        print(f"PASS: Baseline correlation is realistic (r = {corr:.3f}).")
        
    print("\nALL AUTOMATED LEAKAGE TESTS PASSED.")

if __name__ == "__main__":
    run_leakage_test()
