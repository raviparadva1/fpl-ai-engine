import pandas as pd
import json

df = pd.read_csv(r"d:\Fpl_winning\data\processed\phase12_matrix.csv")

# 1. Status Check
print("--- SOLVER STATUS ---")
status_counts = df['status'].value_counts()
print(status_counts)

# 2. Performance by Horizon
print("\n--- PERFORMANCE BY HORIZON ---")
horizons = df.groupby('horizon')[['net_points', 'gross_points', 'transfers', 'hits', 'cap_points']].mean().round(2)
print(horizons)

# 3. Performance by Strategy
print("\n--- PERFORMANCE BY STRATEGY ---")
strategies = df.groupby('strategy')[['net_points', 'gross_points', 'transfers', 'hits', 'cap_points']].mean().round(2)
print(strategies)

# 4. Bench and DGW
print("\n--- BENCH AND DGW PERFORMANCE ---")
extras = df.groupby('horizon')[['bench_points', 'dgw_players_count']].mean().round(2)
print(extras)

# 5. Season Consistency
print("\n--- SEASON CONSISTENCY ---")
seasons = df.groupby(['season', 'horizon'])['net_points'].mean().round(2).unstack()
print(seasons)

# Best combination
best = df.groupby(['horizon', 'strategy'])['net_points'].mean().idxmax()
best_val = df.groupby(['horizon', 'strategy'])['net_points'].mean().max()
print(f"\nRecommended Strategy: Horizon {best[0]}, Mode {best[1]} (Net Pts/GW: {best_val:.2f})")
