import pandas as pd
import os

def audit_data():
    processed_dir = r"d:\Fpl_winning\data\processed"
    df = pd.read_parquet(os.path.join(processed_dir, "cleaned_fpl_data.parquet"))
    
    print("==================================================")
    print("          FPL DATA AUDIT REPORT                   ")
    print("==================================================")
    
    print("\n1. SEASONS AVAILABLE")
    seasons = sorted(df['season'].unique().tolist())
    for s in seasons:
        count = len(df[df['season'] == s])
        print(f"   - {s}: {count:,} observations")
        
    print(f"\n2. PLAYER & OBSERVATION COUNTS")
    print(f"   - Unique Players (by Opta Code): {df['player_code'].nunique():,}")
    print(f"   - Total Player-Gameweek Observations: {len(df):,}")
    
    print("\n3. DUPLICATE RECORDS")
    dupes = df.duplicated(subset=['player_code', 'season', 'GW']).sum()
    print(f"   - Exact duplicates on (player, season, GW): {dupes} (Should be 0 due to DGW aggregation)")
    
    print("\n4. MISSING VALUES (By Column)")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        print(f"   - {col}: {count:,} ({pct:.1f}%)")
        
    print("\n5. AVAILABLE FEATURES")
    print("   " + ", ".join(df.columns.tolist()))
    
    print("\n6. DATA LIMITATIONS")
    print("   - xG/xA/xP (Expected Stats) were introduced officially by FPL midway/recently, hence high missing rates in older seasons.")
    print("   - Double Gameweeks (DGW) are aggregated by (season, GW). Match-level granularity is lost, but point-in-time integrity is preserved.")
    
    print("\n==================================================")

if __name__ == "__main__":
    audit_data()
